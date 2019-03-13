import os
import json
import random

import h5py
import labelme

import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from torch.utils.data import Dataset

class BuildingsDatasetH5(Dataset):

    def __init__(self, img_path):
        super(BuildingsDatasetH5, self).__init__()
        h5_file = h5py.File(img_path)
        self.labels = {
            '_background_': 0,
            'door':1,
            'object':2,
            'wall': 3,
            'window': 4,
        }
        self.data = h5_file.get('image')
        self.target = h5_file.get('label')

    def __getitem__(self, index):
        if index >= len(self): raise IndexError
        img = torch.from_numpy(self.data[index,:,:,:]).float() / 255
        lbl = torch.from_numpy(self.target[index,:,:,:]).long()
        return {'image': img, 'label': lbl}

    def __len__(self):
        return self.data.shape[0]

class BuildingsDataset(Dataset):
    """Buildings dataset."""

    def __init__(self, img_dir, transform=None, transform_seed=None):
        """
        Args:
            img_dir (string): Directory with all the images with labels stored as json.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(BuildingsDataset, self).__init__()

        self.label_name_to_value = {
            '_background_': 0,
            'door':1,
            'object':2,
            'wall': 3,
            'window': 4,
        }
        
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
        #img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #img_labels = {'_background_': 0}
        for shape in sorted(data['shapes'], key=lambda x: x['label']):
            label_name = shape['label']
            if label_name == 'objet':
                shape['label'] = 'object'
                label_name = shape['label']
                
            #if label_name in img_labels:
            label_value = self.label_name_to_value[label_name]
            #else:
            #    label_value = len(img_labels)
            #    img_labels[label_name] = label_value
            #img_labels[label_name] = label_value
            
        #img_labels = {name: self.label_name_to_value[name] for name in img_labels}

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
            img, lbl = self.transform(img, lbl)
            if isinstance(lbl, torch.FloatTensor):
                lbl = (lbl * 255).long()
            sample['image'] = img
            sample['label'] = lbl


        return sample