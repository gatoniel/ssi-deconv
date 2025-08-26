"""The lightning model defining forward pass and loss calculations."""

import torch
from torch import lgamma
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
from ..models.unet import UNet
from ..models.masking import Masking
from ..models.psf_convolution import PSFConvolutionLayer3D


def masked_loss(loss, mask):
    return loss[mask].mean()


class SSIModule(L.LightningModule):
    def __init__(
        self,
        kernel_psf,
        lr=1e-1,
        in_channels=1,
        ndim=3,
        depth: int = 5,
        num_channels_init: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.unet = UNet(
            conv_dims=ndim,
            in_channels=in_channels,
            ndim=ndim,
            depth=depth,
            num_channels_init=num_channels_init,
        )
        self.masked_model = Masking(self.unet, density=0.5)
        self.forward_model = PSFConvolutionLayer3D(kernel_psf, num_channels=in_channels)
        for p in self.forward_model.parameters():
            p.requires_grad = False

    def predict_step(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        img = batch
        y = self.masked_model(img)
        with torch.no_grad():
            x_hat = self.forward_model(y)
        loss = masked_loss(self.loss(x_hat, img), self.model.get_mask())

        self.log("loss", loss, prog_bar=True)
        self.log("masking density", self.masked_loss.density)
        z = img.size()[2] // 2
        tensorboard = self.logger.experiment
        tensorboard.add_image(
            "input",
            img[0, 0, z],
            dataformats="HW",
            global_step=self.global_step,
        )
        tensorboard.add_image(
            "deconv",
            y[0, 0, z],
            dataformats="HW",
            global_step=self.global_step,
        )

        return loss

    def configure_optimizers(self):
        print(self.hparams.lr)
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
