import os
import random

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from facade_project import NUM_IMAGES, NUM_ROTATIONS, FACADE_ROT_IMAGES_TENSORS_DIR, FACADE_ROT_HEATMAPS_TENSORS_DIR


def create_img_to_num_rot(num_img, num_rot_per_img):
    return [num_rot_per_img for _ in range(num_img)]


class FacadeRandomRotDataset(Dataset):
    """
    Facade Random Rotations

    A dataset which return rotated version of an image randomly,
    useful to build batches for data augmentation

    Items of the dataset are: tuple(image, mask) or tuple(image, dict) if add_aux_channels_fn is used

    A demo can be found in "notebook/nb_demo_datasets.ipynb"

    Note that this dataset cannot makes use the CachedDataset directly because it samples images within
    the available rotations. Hence a caching is available directly and implemented here enable
    sampling different rotations
    """

    def __init__(self, img_dir=FACADE_ROT_IMAGES_TENSORS_DIR, add_targets_fn=None, img_to_num_rot=None,
                 caching=False, init_caching=False, device=None):
        Dataset.__init__(self)
        self.dir_path = img_dir
        self.aux_targets_fn = add_targets_fn
        if img_to_num_rot is None:
            img_to_num_rot = create_img_to_num_rot(NUM_IMAGES, NUM_ROTATIONS)
        self.img_to_num_rot = img_to_num_rot
        self.cached_images = None
        self.device = device
        assert not (device is not None and (init_caching or caching)), 'cannot cache on GPU -> GPU_RAM'

        # checking all files exist
        for idx, num_rot in enumerate(self.img_to_num_rot):
            for rot_idx in range(num_rot):
                for is_img in [True, False]:
                    fname = self.get_filename(idx, rot_idx, is_img)
                    assert os.path.isfile(fname), 'file ({}) does not exist'.format(fname)

        if caching or init_caching:
            self.cached_images = dict()
            if init_caching:
                for img_idx in tqdm(list(range(FacadeRandomRotDataset.__len__(self)))):
                    for rot_idx in range(NUM_ROTATIONS):
                        img, lbl = self.get_rot_item(img_idx, rot_idx)
                        self.cached_images[(img_idx, rot_idx)] = (img, lbl)

    def __len__(self):
        return len(self.img_to_num_rot)

    def __getitem__(self, idx):
        rot_idx = random.randint(0, self.img_to_num_rot[idx] - 1)
        if self.cached_images is not None and (idx, rot_idx) in self.cached_images:
            img, lbl = self.cached_images[(idx, rot_idx)]
        else:
            if idx >= FacadeRandomRotDataset.__len__(self): raise IndexError
            img, lbl = self.get_rot_item(idx, rot_idx)
            if self.cached_images is not None:
                self.cached_images[(idx, rot_idx)] = (img, lbl)
        return img, lbl

    def get_rot_item(self, idx, rot_idx):
        img, lbl = torch.load(self.get_filename(idx, rot_idx, True), map_location=self.device), \
                   torch.load(self.get_filename(idx, rot_idx, False), map_location=self.device).long()
        targets = lbl
        if self.aux_targets_fn is not None:
            targets = {
                'mask': lbl,
            }
            aux_targets = self.aux_targets_fn(idx, rot_idx, device=self.device)
            assert type(aux_targets) is dict
            targets.update(aux_targets)
        return img, targets

    def get_filename(self, img_idx, rot_idx, is_img):
        name = 'img' if is_img else 'lbl'
        return '{}/{}_{:03d}_{:03d}.torch'.format(self.dir_path, name, img_idx, rot_idx)


def add_heatmaps_target(img_idx, rot_idx, device):
    def get_filename(idx, jdx):
        return '{}/heatmaps_door-window_{:03d}_{:03d}.torch' \
            .format(FACADE_ROT_HEATMAPS_TENSORS_DIR, idx, jdx)

    return {
        'heatmaps': torch.load(get_filename(img_idx, rot_idx), map_location=device)
    }
