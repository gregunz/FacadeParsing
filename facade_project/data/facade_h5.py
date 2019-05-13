import h5py
import torch
from torch.utils.data import Dataset
from torchvision.utils import make_grid

from facade_project import LABEL_NAME_TO_VALUE


class FacadesDatasetH5Patches(Dataset):
    def __init__(self, img_path):
        Dataset.__init__(self)
        h5_file = h5py.File(img_path, 'r')
        self.labels = LABEL_NAME_TO_VALUE
        self.data = h5_file.get('image')
        self.target = h5_file.get('label')

    def __getitem__(self, index):
        if index >= FacadesDatasetH5Patches.__len__(self): raise IndexError
        img = torch.from_numpy(self.data[index, :, :, :]).float() / 255
        lbl = torch.from_numpy(self.target[index, :, :, :]).long()
        return img, lbl

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

        img = torch.stack([img for img, _ in data])
        img = make_grid(img, nrow=self.n_rows[index], padding=0)

        lbl = torch.stack([lbl for _, lbl in data])
        lbl = make_grid(lbl, nrow=self.n_rows[index], padding=0)
        lbl = lbl[0:1]  # because make_grid return 3 (identical) channels for the image

        return img, lbl

    def __len__(self):
        return len(self.n_crops)
