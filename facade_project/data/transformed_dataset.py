from torch.utils.data import Dataset


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, lbl = self.transform(*self.dataset.__getitem__(index))
        return img, lbl

    def __len__(self):
        return len(self.dataset)