import random

import numpy as np
from torch.utils.data import Subset, DataLoader

from facade_project import DEFAULT_SEED_SPLIT
from facade_project.data.cached_dataset import CachedDataset
from facade_project.data.facade_dataset_labelme import FacadeDatasetLabelme
from facade_project.data.facade_heatmap_dataset import FacadeHeatmapDataset
from facade_project.data.facade_random_rot_dataset import FacadeDatasetRandomRot
from facade_project.data.transformed_dataset import TransformedDataset


def split(dataset, seed=DEFAULT_SEED_SPLIT):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    def lenghts_split(size, perc):
        train_l = int(perc * size)
        return train_l, size - train_l

    num_images = len(dataset)
    n_img_train, n_img_val = lenghts_split(num_images, 0.9)

    img_indices = np.random.permutation(num_images)
    train_ind = img_indices[:n_img_train]
    val_ind = img_indices[n_img_train:]

    assert len(val_ind) == n_img_val
    len(train_ind), len(val_ind)

    return Subset(dataset, train_ind), Subset(dataset, val_ind)


def to_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
