import torch
from torch import nn


class PSFConvolutionLayer3D(nn.Module):
    def __init__(self, kernel_psf, num_channels=1):
        super().__init__()
        paddings = [(s - 1) // 2 for s in kernel_psf.shape]
        self.seq = nn.Sequential(
            nn.ReplicationPad3d(tuple(paddings[i] for i in [2, 2, 1, 1, 0, 0])),
            nn.Conv3d(
                num_channels,
                num_channels,
                kernel_psf.shape,
                stride=1,
                padding=0,
                bias=False,
                groups=num_channels,
            ),
        )

        self.weights_init(kernel_psf)

    def weights_init(self, kernel_psf):
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(kernel_psf))

    def forward(self, x):
        return self.seq(x)
