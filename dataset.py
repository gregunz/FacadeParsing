import os
import json
import random

import cv2
import labelme
import skimage

import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class BuildingsDataset(Dataset):
    """Buildings dataset."""

    def __init__(self, img_dir, transform=None, transform_seed=None):
        """
        Args:
            img_dir (string): Directory with all the images with labels stored as json.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.label_name_to_value = {
            '_background_': 0,
            'door':1,
            'object':2,
            'wall': 3,
            'window': 4,
        }
        
        def json_contain_image(path):
            data = json.load(open(path))
            return data['imageData']
        
        self.img_paths = [os.path.join(img_dir, fname) for fname in sorted(os.listdir(img_dir))]
        self.img_paths = [path for path in self.img_paths if json_contain_image(path)]

        def transform_tuple(image, label):
            if transform:
                seed = transform_seed
                if not transform_seed:
                    seed = np.random.randint(2147483647)
                random.seed(seed)
                image = transform(image)
                random.seed(seed)
                label = transform(label)
            return {'image': image, 'label': label}

        self.transform = transform_tuple

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        data = json.load(open(img_path))

        imageData = data['imageData']
        img = labelme.utils.img_b64_to_arr(imageData)
        #img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #img_labels = {'_background_': 0}

        for shape in sorted(data['shapes'], key=lambda x: x['label']):
            label_name = shape['label']
            if label_name == 'objet':
                shape['label'] = 'object'
            if label_name in self.label_name_to_value:
                label_value = self.label_name_to_value[label_name]
            else:
                label_value = len(self.label_name_to_value)
                self.label_name_to_value[label_name] = label_value
            #img_labels[label_name] = label_value

        try:
            lbl = labelme.utils.shapes_to_label(img.shape, data['shapes'], self.label_name_to_value)
        except AssertionError:
            print("ERROR occured while trying to construct labels of " + img_path)
            lbl = np.zeros_like(img, dtype='uint8')

        # int to float
        img = (img / 255).astype('float32')
        #img = F.to_pil_image(img)
        # int32 to uint8
        lbl = lbl.astype('uint8')[:, :, np.newaxis]
                
        sample = {'image': img, 'label': lbl}#, 'label_names': img_labels}
        if self.transform:
            sample = self.transform(**sample)
            if isinstance(sample['label'], torch.FloatTensor):
                sample['label'] = (sample['label'] * 255).long()


        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        sample = sample.copy()
        sample['image'] = torch.from_numpy(image)
        sample['label'] = torch.from_numpy(label)
        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        sample = sample.copy()
        sample['image'] = image
        sample['label'] = label
        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = skimage.transform.resize(image, (new_h, new_w))
        lbl = skimage.transform.resize(label, (new_h, new_w), preserve_range=True).astype('uint8')

        sample = sample.copy()
        sample['image'] = img
        sample['label'] = lbl
        return sample
