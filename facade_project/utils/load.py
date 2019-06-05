import json

import labelme
import numpy as np
import torch
from labelme.utils import img_b64_to_arr

from facade_project import LABEL_NAME_TO_VALUE, NUM_IMAGES, HEATMAP_TYPES, IS_SIGMA_FIXED, SIGMA_FIXED, SIGMA_SCALE, \
    FACADE_ROT_IMAGES_TENSORS_DIR, FACADE_ROT_HEATMAPS_INFOS_PATH, HEATMAP_LABELS
from facade_project.geometry.heatmap import points_to_cwh, build_heatmaps


def load_infos_per_rot(path):
    all_infos = json.load(open(path, mode='r'))
    return {
        int(k): {
            int(k2): v2 for k2, v2 in info_for_each_rot.items()
        } for k, info_for_each_rot in all_infos.items()
    }


HEATMAP_INFOS_PER_ROT = load_infos_per_rot(path=FACADE_ROT_HEATMAPS_INFOS_PATH)

# 000.torch because they are not rotated but only resized
IMG_000_PATHS = ['{}/img_{:03d}_000.torch'.format(FACADE_ROT_IMAGES_TENSORS_DIR, i) for i in range(NUM_IMAGES)]
HEATMAPS_000_INFOS = [HEATMAP_INFOS_PER_ROT[i][0] for i in range(len(IMG_000_PATHS))]


def load_tuple_from_json(img_path, label_name_to_value=LABEL_NAME_TO_VALUE):
    data = json.load(open(img_path))

    image_data = data['imageData']
    img = labelme.utils.img_b64_to_arr(image_data)

    data['shapes'] = [shape for shape in data['shapes'] if shape['label'] in label_name_to_value]

    lbl = labelme.utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
    lbl = lbl.astype('uint8')[:, :, np.newaxis]

    return img, lbl


def load_heatmaps_info(labelme_json_path):
    json_data = json.load(open(labelme_json_path, mode='r'))
    return extract_heatmaps_info(json_data)


def extract_heatmaps_info(json_data, label_name_to_value=LABEL_NAME_TO_VALUE):
    img = img_b64_to_arr(json_data['imageData'])
    info = {
        'img_height': img.shape[0],
        'img_width': img.shape[1],
        'cwh_list': [],
    }
    for shape in json_data['shapes']:
        lbl = shape['label']
        if lbl in label_name_to_value:
            points = shape['points']
            if len(points) > 3:
                c_x, c_y, w, h = points_to_cwh(points)
                info['cwh_list'].append({
                    'label': lbl,
                    'center': (c_x, c_y),
                    'width': w,
                    'height': h,
                })
    return info

def load_img_heatmaps(
        index,
        img_tensor_paths=IMG_000_PATHS,
        heatmap_infos=HEATMAPS_000_INFOS,
        labels=HEATMAP_LABELS,
        is_sigma_fixed=IS_SIGMA_FIXED,
        sigma_fixed=SIGMA_FIXED,
        sigma_scale=SIGMA_SCALE,
        heatmap_types=HEATMAP_TYPES,
):
    img = torch.load(img_tensor_paths[index])
    heatmaps = build_heatmaps(
        heatmap_info=heatmap_infos[index],
        labels=labels,
        heatmap_types=heatmap_types,
        is_sigma_fixed=is_sigma_fixed,
        sigma_fixed=sigma_fixed,
        sigma_scale=sigma_scale,
    )
    return img, heatmaps
