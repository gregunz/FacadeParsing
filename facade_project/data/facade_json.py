import json
import os

import labelme
import numpy as np
from torch.utils.data import Dataset

from facade_project import LABEL_NAME_TO_VALUE


class FacadesDatasetJson(Dataset):
    """Buildings dataset."""

    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images with labels stored as json.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        Dataset.__init__(self)

        self.label_name_to_value = LABEL_NAME_TO_VALUE

        self.img_paths = [os.path.join(img_dir, filename) for filename in sorted(os.listdir(img_dir))]
        self.img_paths = [path for path in self.img_paths]
        self.transform = transform

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

        if self.transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl
