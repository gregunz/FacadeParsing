from torch.utils.data import Dataset

from facade_project import LABEL_NAME_TO_VALUE
from facade_project.utils.load import IMG_RESIZED_TENSOR_PATHS, HEATMAP_INFOS, load_img_heatmaps


class FacadeHeatmapDataset(Dataset):
    """
    Facade Heatmap Dataset
    """

    def __init__(self, img_tensor_paths=IMG_RESIZED_TENSOR_PATHS, heatmap_infos=HEATMAP_INFOS,
                 label_name_to_value=LABEL_NAME_TO_VALUE):
        Dataset.__init__(self)
        self.label_name_to_value = label_name_to_value
        self.img_tensor_paths = img_tensor_paths
        self.heatmap_infos = heatmap_infos

    def __getitem__(self, index):
        return load_img_heatmaps(index, self.img_tensor_paths, self.heatmap_infos)

    def __len__(self):
        return len(self.img_tensor_paths)
