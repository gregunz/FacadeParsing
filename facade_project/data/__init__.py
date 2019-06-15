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


def get_indices_split(length, seed=DEFAULT_SEED_SPLIT, percentage=0.9):
    """
    Create two list of indices given a range length
    :param length: length of the range
    :param seed: the seed to split the indices
    :param percentage: the percentage of the first indices list (rest will be to the right indices list)
    :return: tuple of indices, (left, right) or (training, validating)
    """
    assert 0 <= percentage <= 1
    assert 0 < length

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_left = int(length * percentage)
    img_indices = np.random.permutation(length)
    return img_indices[:n_left], img_indices[n_left:]


def split(dataset, seed=DEFAULT_SEED_SPLIT, percentage=0.9):
    """
    Splits dataset into two datasets for training and testing/validating purpose
    :param dataset: the dataset to split
    :param seed: the seed to split the data
    :param percentage: the percentage of the training data (rest will be testing/validating data)
    :return: tuple of dataset, (train, test/val)
    """
    train_ind, val_ind = get_indices_split(len(dataset), seed, percentage)
    return Subset(dataset, train_ind), Subset(dataset, val_ind)


def to_dataloader(dataset, batch_size, shuffle=True):
    """
    Turn a dataset into a shuffled DataLoader. It handles dict and HeatmapsInfo type.

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

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
