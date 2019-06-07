import json

import torch
from torch.utils.data import Dataset

from facade_project import LABEL_NAME_TO_VALUE, SIGMA_FIXED, IS_SIGMA_FIXED, SIGMA_SCALE, \
    HEATMAP_TYPES_HANDLED, HEATMAP_LABELS, FACADE_ROT_HEATMAPS_INFOS_PATH, FACADE_ROT_IMAGES_TENSORS_DIR, NUM_IMAGES
from facade_project.geometry.heatmap import build_heatmaps


class FacadeHeatmapDataset(Dataset):
    """
    Facade Heatmap Dataset

    A dataset of images and heatmaps representing where windows and door are located
    Heatmaps are divided into 3 sub-heatmaps (channels),
    one for the locations, one for the width, and one for the height

    Items of the dataset are: tuple(image, heatmaps)

    A demo can be found in "notebook/nb_demo_datasets.ipynb"
    """

    def __init__(
            self,
            img_tensor_paths=None,
            heatmap_infos=None,
            label_name_to_value=LABEL_NAME_TO_VALUE,
            is_sigma_fixed=IS_SIGMA_FIXED,
            sigma_fixed=SIGMA_FIXED,
            sigma_scale=SIGMA_SCALE,
            heatmap_types=HEATMAP_TYPES_HANDLED,
            heatmap_labels=HEATMAP_LABELS,
    ):
        Dataset.__init__(self)
        if img_tensor_paths is None:
            img_tensor_paths = IMG_000_PATHS
        if heatmap_infos is None:
            heatmap_infos = HEATMAPS_000_INFOS
        assert len(img_tensor_paths) == len(heatmap_infos)

        self.img_tensor_paths = img_tensor_paths
        self.heatmap_infos = heatmap_infos
        self.label_name_to_value = label_name_to_value
        self.is_sigma_fixed = is_sigma_fixed
        self.sigma_fixed = sigma_fixed
        self.sigma_scale = sigma_scale
        self.heatmap_types = heatmap_types
        self.heatmap_labels = heatmap_labels

    def __getitem__(self, index):
        img = torch.load(self.img_tensor_paths[index])
        heatmaps = build_heatmaps(
            heatmap_info=self.heatmap_infos[index],
            labels=self.heatmap_labels,
            heatmap_types=self.heatmap_types,
            is_sigma_fixed=self.is_sigma_fixed,
            sigma_fixed=self.sigma_fixed,
            sigma_scale=self.sigma_scale,
        )
        return img, heatmaps

    def __len__(self):
        return len(self.img_tensor_paths)


def __load_infos_per_rot__(path):
    all_infos = json.load(open(path, mode='r'))
    heatmaps_infos = {
        int(k): {
            int(k2): v2 for k2, v2 in info_for_each_rot.items()
        } for k, info_for_each_rot in all_infos.items()
    }
    for infos_per_rot in heatmaps_infos.values():
        for info in infos_per_rot.values():
            info['cwh_list'] = [cwh for cwh in info['cwh_list'] \
                                if 0 <= cwh['center'][0] <= info['img_width'] and 0 <= cwh['center'][1] <= info[
                                    'img_height']]
    return heatmaps_infos


HEATMAP_INFOS_PER_ROT = __load_infos_per_rot__(path=FACADE_ROT_HEATMAPS_INFOS_PATH)
# 000.torch because we only take the non rotated ones for this dataset
IMG_000_PATHS = ['{}/img_{:03d}_000.torch'.format(FACADE_ROT_IMAGES_TENSORS_DIR, i) for i in range(NUM_IMAGES)]
HEATMAPS_000_INFOS = [HEATMAP_INFOS_PER_ROT[i][0] for i in range(len(IMG_000_PATHS))]
