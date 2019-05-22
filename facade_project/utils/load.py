import json
import pickle

import labelme
import numpy as np
import torch
from labelme.utils import img_b64_to_arr

from facade_project import LABEL_NAME_TO_VALUE, PATH_TO_DATA, NUM_IMAGES, IMG_MAX_SIZE, HEATMAP_INCLUDE_MASK
from facade_project.geometry.heatmap import points_to_cwh, build_heatmaps
from facade_project.geometry.image import resize

# 000.torch because they are not rotated but only resized
IMG_RESIZED_TENSOR_PATHS = ['{}/images/rot_aug_{}/img_{:03d}_000.torch'.format(PATH_TO_DATA, IMG_MAX_SIZE, i) \
                            for i in range(NUM_IMAGES)]
LBL_RESIZED_TENSORS_PATH = [p.replace('img_', 'lbl_') for p in IMG_RESIZED_TENSOR_PATHS]
RESIZED_INFOS = pickle.load(open('{}/images/rot_aug_{}/images_infos.p'.format(PATH_TO_DATA, IMG_MAX_SIZE), mode='rb'))

HEATMAP_INFOS = pickle.load(open('{}/heatmaps/heatmaps_infos.p'.format(PATH_TO_DATA), mode='rb'))


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
        img_tensor_paths=IMG_RESIZED_TENSOR_PATHS,
        heatmap_infos=HEATMAP_INFOS,
        mask_tensor_paths=LBL_RESIZED_TENSORS_PATH,
        include_mask=HEATMAP_INCLUDE_MASK,
        max_size=None,
):
    img = torch.load(img_tensor_paths[index])
    if max_size is None:
        max_size = max(img.shape[1:])
    cropped_bbox = RESIZED_INFOS[(index, 0)]['cropped_bbox']
    heatmaps = build_heatmaps(heatmap_infos[index], max_size=max_size, cropped_bbox=cropped_bbox)
    if include_mask:
        mask = torch.load(mask_tensor_paths[index])
        mask = resize(mask, max_size=max_size)
        mask = mask.float()  # same type as heatmaps
        heatmaps = torch.cat([heatmaps, mask], dim=0)
    return img, heatmaps
