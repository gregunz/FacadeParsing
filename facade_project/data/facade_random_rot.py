import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm_notebook as tqdm

from facade_project import NUM_IMAGES, NUM_ROTATIONS


class FacadesDatasetRandomRot(Dataset):
    def __init__(self, img_dir, caching=False, init_caching=False):
        Dataset.__init__(self)
        self.dir_path = img_dir
        self.num_images = NUM_IMAGES  # hardcoded from constants
        self.num_rotations = NUM_ROTATIONS  # hardcoded from constants
        self.cached_images = None
        if caching or init_caching:
            self.cached_images = dict()
            if init_caching:
                for img_idx in tqdm(list(range(FacadesDatasetRandomRot.__len__(self)))):
                    for rot_idx in range(NUM_ROTATIONS):
                        img, lbl = torch.load(self.get_filename(img_idx, rot_idx, True)), \
                                   torch.load(self.get_filename(img_idx, rot_idx, False)).long()
                        self.cached_images[(img_idx, rot_idx)] = (img, lbl)

    def __len__(self):
        return NUM_IMAGES

    def get_filename(self, img_idx, rot_idx, is_img):
        name = 'img' if is_img else 'lbl'
        return '{}/{}_{:03d}_{:03d}.torch'.format(self.dir_path, name, img_idx, rot_idx)

    def __getitem__(self, idx):
        rot_idx = random.randint(0, self.num_rotations - 1)
        if self.cached_images is not None and (idx, rot_idx) in self.cached_images:
            img, lbl = self.cached_images[(idx, rot_idx)]
        else:
            if idx >= FacadesDatasetRandomRot.__len__(self): raise IndexError
            img, lbl = torch.load(self.get_filename(idx, rot_idx, True)), \
                       torch.load(self.get_filename(idx, rot_idx, False))
            if self.cached_images is not None:
                self.cached_images[(idx, rot_idx)] = (img, lbl)
        return img, lbl
