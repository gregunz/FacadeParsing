import os
import json
import random

import h5py
import labelme

import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision.transforms import Normalize

import torchvision.transforms.functional as F

from torch.utils.data import Dataset

from constants import label_name_to_value, mean, std, num_images, num_rotations

class FacadesDatasetRandomRot(Dataset):
    def __init__(self, img_dir, transform=None):
        Dataset.__init__(self)
        self.dir_path = img_dir
        self.num_images = num_images # hardcoded from constants
        self.num_rotations = num_rotations # hardcoded from constants
        self.transform = transform

    def __len__(self):
        return num_images
    
    def get_filename(self, img_idx, rot_idx, is_img):
        name = 'img' if is_img else 'lbl'
        return '{}/{}_{:03d}_{:03d}.torch'.format(self.dir_path, name, img_idx, rot_idx)

    def __getitem__(self, idx):
        if idx >= FacadesDatasetRandomRot.__len__(self): raise IndexError
        rot_idx = random.randint(0, self.num_rotations - 1)
        img, lbl = torch.load(self.get_filename(idx, rot_idx, True)),\
               torch.load(self.get_filename(idx, rot_idx, False))
        if self.transform is not None:
            img, lbl = self.transform(img, lbl)
        return img, lbl

class FacadesDatasetH5Patches(Dataset):
    def __init__(self, img_path, normalized=False):
        Dataset.__init__(self)
        h5_file = h5py.File(img_path, 'r')
        self.normalized = normalized
        self.labels = label_name_to_value
        self.data = h5_file.get('image')
        self.target = h5_file.get('label')

    def __getitem__(self, index):
        if index >= FacadesDatasetH5Patches.__len__(self): raise IndexError
        img = torch.from_numpy(self.data[index,:,:,:]).float() / 255
        if self.normalized:
            img = Normalize(mean=mean, std=std)(img)
        lbl = torch.from_numpy(self.target[index,:,:,:]).long()
        return {'image': img, 'label': lbl}

    def __len__(self):
        return self.data.shape[0]


class FacadesDatasetH5Full(FacadesDatasetH5Patches):
    def __init__(self, img_path, n_rows, n_crops, normalized=False):
        FacadesDatasetH5Patches.__init__(self, img_path, normalized)
        assert len(n_rows) == len(n_crops)
        assert sum(n_crops) == self.data.shape[0]
        self.n_rows = n_rows
        self.n_crops = n_crops
        
    def __getitem__(self, index):
        if index >= FacadesDatasetH5Full.__len__(self): raise IndexError
        patch_index = sum(self.n_crops[:index])
        n_crop = self.n_crops[index]
        get = lambda idx: FacadesDatasetH5Patches.__getitem__(self, idx)
        data = [get(patch_index + i) for i in range(n_crop)]
        
        img = torch.stack([d['image'] for d in data])
        img = make_grid(img, nrow=self.n_rows[index], padding=0)
        
        lbl = torch.stack([d['label'] for d in data])
        lbl = make_grid(lbl, nrow=self.n_rows[index], padding=0)
        lbl = lbl[0:1] #because make_grid return 3 (identical) channels for the image
        
        return {'image': img, 'label': lbl}
    
    def __len__(self):
        return len(self.n_crops)


class FacadesDatasetJson(Dataset):
    """Buildings dataset."""

    def __init__(self, img_dir, transform=None, transform_seed=None):
        """
        Args:
            img_dir (string): Directory with all the images with labels stored as json.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        Dataset.__init__(self)

        self.label_name_to_value = label_name_to_value
        
        self.img_paths = [os.path.join(img_dir, fname) for fname in sorted(os.listdir(img_dir))]
        self.img_paths = [path for path in self.img_paths]

        def transform_tuple(image, label):
            if transform:
                seed = transform_seed
                if not transform_seed:
                    seed = np.random.randint(2147483647)
                random.seed(seed)
                image = transform(image)
                random.seed(seed)
                label = transform(label)
            return image, label

        self.transform = transform_tuple

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        data = json.load(open(img_path))

        imageData = data['imageData']
        img = labelme.utils.img_b64_to_arr(imageData)

        """
        for shape in sorted(data['shapes'], key=lambda x: x['label']):
            label_name = shape['label']
            if label_name == 'objet':
                shape['label'] = 'object'
                label_name = shape['label']
                
            label_value = self.label_name_to_value[label_name]
        """
        # removing object (and misnamed objet) class
        data['shapes'] = [shape for shape in data['shapes'] if shape['label'] != 'object' and shape['label'] != 'objet']

        try:
            lbl = labelme.utils.shapes_to_label(img.shape, data['shapes'], self.label_name_to_value)
        except AssertionError:
            print("ERROR occured while trying to construct labels of " + img_path)
            lbl = np.zeros_like(img, dtype='uint8')

        # int to float
        img = (img / 255).astype('float32')
        # int32 to uint8
        lbl = lbl.astype('uint8')[:, :, np.newaxis]
                
        sample = {'image': img, 'label': lbl}#, 'label_names': img_labels}
        if self.transform:
            img, lbl = self.transform(img, lbl)
            if isinstance(lbl, torch.FloatTensor):
                lbl = (lbl * 255).long()
            sample['image'] = img
            sample['label'] = lbl


        return sample