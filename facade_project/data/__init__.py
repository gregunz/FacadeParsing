import random

import numpy as np
from torch.utils.data import Subset, DataLoader
from torch.utils.data.dataloader import default_collate

from facade_project import DEFAULT_SEED_SPLIT
from facade_project.data.cached_dataset import CachedDataset
from facade_project.data.facade_heatmap_dataset import FacadeHeatmapDataset
from facade_project.data.facade_labelme_dataset import FacadeLabelmeDataset
from facade_project.data.facade_random_rot_dataset import FacadeRandomRotDataset
from facade_project.data.transformed_dataset import TransformedDataset


def split(dataset, seed=DEFAULT_SEED_SPLIT, percentage=0.9):
    """
    Splits dataset into two datasets for training and testing/validating purpose
    :param dataset: the dataset to split
    :param seed: the seed to split the data
    :param percentage: the percentage of the training data (rest will be testing/validating data)
    :return: tuple of dataset, (train, test/val)
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    def lenghts_split(size, perc):
        train_l = int(perc * size)
        return train_l, size - train_l

    num_images = len(dataset)
    n_img_train, n_img_val = lenghts_split(num_images, percentage)

    img_indices = np.random.permutation(num_images)
    train_ind = img_indices[:n_img_train]
    val_ind = img_indices[n_img_train:]

    len(train_ind), len(val_ind)
    return Subset(dataset, train_ind), Subset(dataset, val_ind)


def to_dataloader(dataset, batch_size):
    """
    Turn a dataset into a shuffled DataLoader
    :param dataset: the dataset
    :param batch_size: size of a batch
    :return: a DataLoader
    """

    def collate_fn(batch):
        if type(batch[0][1]) is dict and 'heatmaps_info' in batch[0][1]:
            heatmaps_infos = []
            for inputs, targets in batch:
                heatmaps_infos.append(targets['heatmaps_info'])
                del targets['heatmaps_info']

            batch_collated = default_collate(batch)
            batch_collated[1]['heatmaps_info'] = heatmaps_infos
            return batch_collated
        else:
            return default_collate(batch)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
