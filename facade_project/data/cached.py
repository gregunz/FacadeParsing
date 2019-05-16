from torch.utils.data import Dataset
from tqdm.auto import tqdm


class CachedDataset(Dataset):
    def __init__(self, dataset, init_caching=False):
        self.dataset = dataset
        self.cache = dict()
        if init_caching:
            for idx, data in enumerate(tqdm(self.dataset)):
                self.cache[idx] = data

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        data = self.dataset[index]
        self.cache[index] = data
        return data

    def __len__(self):
        return len(self.dataset)
