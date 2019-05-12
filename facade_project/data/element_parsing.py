import os

from torch.utils.data import Dataset

from facade_project import LABEL_NAME_TO_VALUE


class FacadeCWHDataset(Dataset):
    """
    Facade Center-Width-Height Dataset
    """
    def __init__(self, json_dir):
        Dataset.__init__(self)
        self.label_name_to_value = LABEL_NAME_TO_VALUE
        self.img_paths = [os.path.join(json_dir, filename) for filename in sorted(os.listdir(json_dir))]
        self.img_paths = [path for path in self.img_paths]

    def __getitem__(self, index):
        return None

    def __len__(self):
        return len(self.img_paths)