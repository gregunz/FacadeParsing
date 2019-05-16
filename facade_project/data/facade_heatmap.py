from torch.utils.data import Dataset

from facade_project import LABEL_NAME_TO_VALUE
from facade_project.utils.load import IMG_RESIZED_TENSORS_PATH, HEATMAP_INFOS, load_img_heatmaps


class FacadeHeatmapDataset(Dataset):
    """
    Facade Heatmap Dataset
    """

    def __init__(self, img_tensors_paths=IMG_RESIZED_TENSORS_PATH, heatmap_infos=HEATMAP_INFOS,
                 label_name_to_value=LABEL_NAME_TO_VALUE):
        Dataset.__init__(self)
        self.label_name_to_value = label_name_to_value
        self.img_tensors_paths = img_tensors_paths
        self.heatmap_infos = heatmap_infos

    def __getitem__(self, index):
        return load_img_heatmaps(index, self.img_tensors_paths, self.heatmap_infos)

    def __len__(self):
        return len(self.img_tensors_paths)
