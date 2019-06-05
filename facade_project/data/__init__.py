import random

import numpy as np
from torch.utils.data import Subset, DataLoader

from facade_project import DEFAULT_SEED_SPLIT
from facade_project.data.cached_dataset import CachedDataset
from facade_project.data.facade_heatmap_dataset import FacadeHeatmapDataset
from facade_project.data.facade_labelme_dataset import FacadeLabelmeDataset
from facade_project.data.facade_random_rot_dataset import FacadeRandomRotDataset
from facade_project.data.transformed_dataset import TransformedDataset


def split(dataset, seed=DEFAULT_SEED_SPLIT):
    if seed == DEFAULT_SEED_SPLIT:
        # This was done manually for the following reasons:
        # -> duplicates
        # -> bad/missing labels
        val_ind = [6, 9, 14, 22, 24, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 45, 47, 51, 58, 60, 61, 64, 85, 86, 95,
                   104, 107, 112, 125, 127, 130, 131, 142, 145, 150, 162, 163, 165, 166, 167, 168, 183, 187, 188, 194,
                   201, 226, 228, 273, 274, 287, 288, 295, 318, 325, 328, 329, 347, 348, 372, 376, 386, 391, 408, 409,
                   411, 415, 416, 417]
        train_ind = list(set(range(418)) - set(val_ind) - {20, 23, 25, 28, 29, 106, 361})
    else:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        print("WARNING: non optimal split was done, see above code comments for more information")

        def lenghts_split(size, perc):
            train_l = int(perc * size)
            return train_l, size - train_l

        num_images = len(dataset)
        n_img_train, n_img_val = lenghts_split(num_images, 0.9)

        img_indices = np.random.permutation(num_images)
        train_ind = img_indices[:n_img_train]
        val_ind = img_indices[n_img_train:]

    len(train_ind), len(val_ind)
    return Subset(dataset, train_ind), Subset(dataset, val_ind)


def to_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
