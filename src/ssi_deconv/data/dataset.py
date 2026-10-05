"""Dataset class to organize training preparation."""

from itertools import product

import numpy as np
from skimage.exposure import rescale_intensity
from torch.utils.data import Dataset


def prepare_patches(imgs, patch_size):
    patches = []
    for i, img in enumerate(imgs):
        assert img.ndim == patch_size.ndim + 1
        slices = tuple([] for _ in range(patch_size.ndim))
        for j in range(patch_size.ndim):
            startpoints = np.arange(
                0, img.shape[j + 1] - patch_size[j] + 1, patch_size[j] // 2
            )
            for start in startpoints:
                slices[j].append(slice(start, start + patch_size[j]))
        patches.extend([(i, sl) for sl in product(*slices)])
    return patches


class SSIDataset(Dataset):
    def __init__(
        self,
        imgs: list[np.ndarray],
        patch_size: tuple[int, ...],
        normalize: bool = True,
    ):
        if normalize:
            imgs = [
                rescale_intensity(
                    img.astype(np.float32), in_range="image", out_range=(0, 1)
                )
                for img in imgs
            ]
        # TODO: allow for arbitrary input channels
        self.imgs = [img[np.newaxis, ...] for img in imgs]
        self.patches = prepare_patches(imgs, patch_size)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        i, sl = self.patches[idx]
        return self.imgs[i][(slice(None)) + sl]
