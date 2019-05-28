from torch.utils.data import Dataset

from facade_project import LABEL_NAME_TO_VALUE, HEATMAP_INCLUDE_MASK, SIGMA_FIXED, IS_SIGMA_FIXED, SIGMA_SCALE, \
    HEATMAP_TYPES
from facade_project.utils.load import IMG_RESIZED_TENSOR_PATHS, HEATMAP_INFOS, load_img_heatmaps, \
    LBL_RESIZED_TENSORS_PATH


class FacadeHeatmapDataset(Dataset):
    """
    Facade Heatmap Dataset
    """

    def __init__(
            self,
            img_tensor_paths=IMG_RESIZED_TENSOR_PATHS,
            heatmap_infos=HEATMAP_INFOS,
            mask_tensor_paths=LBL_RESIZED_TENSORS_PATH,
            include_mask=HEATMAP_INCLUDE_MASK,
            size=None,
            label_name_to_value=LABEL_NAME_TO_VALUE,
            is_sigma_fixed=IS_SIGMA_FIXED,
            sigma_fixed=SIGMA_FIXED,
            sigma_scale=SIGMA_SCALE,
            heatmap_types=HEATMAP_TYPES

    ):
        Dataset.__init__(self)
        self.img_tensor_paths = img_tensor_paths
        self.heatmap_infos = heatmap_infos
        self.mask_tensor_paths = mask_tensor_paths
        self.include_mask = include_mask
        self.size = size
        self.label_name_to_value = label_name_to_value
        self.is_sigma_fixed = is_sigma_fixed
        self.sigma_fixed = sigma_fixed
        self.sigma_scale = sigma_scale
        self.heatmap_types = heatmap_types

    def __getitem__(self, index):
        return load_img_heatmaps(
            index=index,
            img_tensor_paths=self.img_tensor_paths,
            heatmap_infos=self.heatmap_infos,
            mask_tensor_paths=self.mask_tensor_paths,
            include_mask=self.include_mask,
            size=self.size,
            label_name_to_value=self.label_name_to_value,
            is_sigma_fixed=self.is_sigma_fixed,
            sigma_fixed=self.sigma_fixed,
            sigma_scale=self.sigma_scale,
            heatmap_types=self.heatmap_types,
        )

    def __len__(self):
        return len(self.img_tensor_paths)
