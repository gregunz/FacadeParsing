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

DEFAULT_VAL_IND = [6, 9, 22, 24, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 45, 47, 51, 58, 60, 61, 64, 85, 86, 95,
                   104, 107, 112, 125, 127, 130, 131, 142, 145, 150, 162, 163, 165, 166, 167, 168, 183, 187, 188, 194,
                   201, 226, 228, 273, 274, 287, 288, 295, 318, 325, 328, 329, 347, 348, 372, 376, 386, 391, 408, 409,
                   411, 415, 416, 417]
DEFAULT_IGNORED_IND = [20, 23, 25, 28, 29, 106, 361]
DEFAULT_TRAIN_IND = list(set(range(418)) - set(DEFAULT_VAL_IND) - set(DEFAULT_IGNORED_IND))


def split(dataset, seed=DEFAULT_SEED_SPLIT, percentage=0.9):
    """
    Splits dataset into two datasets for training and testing/validating purpose
    :param dataset: the dataset to split
    :param seed: the seed to split the data
    :param percentage: the percentage of the training data (rest will be testing/validating data)
    :return: tuple of dataset, (train, test/val)
    """
    if seed == DEFAULT_SEED_SPLIT:
        assert len(dataset) == 418
        # This was done manually/hardcoded for the following reasons:
        # -> duplicates
        # -> bad/missing labels
        train_ind = DEFAULT_TRAIN_IND
        val_ind = DEFAULT_VAL_IND
    else:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        print("WARNING: non optimal split was done, see above code comments for more information")

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
