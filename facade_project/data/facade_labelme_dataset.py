import os

from torch.utils.data import Dataset

from facade_project import LABEL_NAME_TO_VALUE
from facade_project.utils.load import load_tuple_from_json


class FacadeLabelmeDataset(Dataset):
    """Buildings dataset."""

    def __init__(self, img_dir, label_name_to_value=LABEL_NAME_TO_VALUE):
        Dataset.__init__(self)

        self.label_name_to_value = label_name_to_value

        self.img_paths = [os.path.join(img_dir, filename) for filename in sorted(os.listdir(img_dir))]
        self.img_paths = [path for path in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return load_tuple_from_json(img_path, self.label_name_to_value)
