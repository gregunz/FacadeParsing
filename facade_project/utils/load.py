import json
import pickle

import labelme
import numpy as np
import torch
from labelme.utils import img_b64_to_arr

from facade_project import LABEL_NAME_TO_VALUE, PATH_TO_DATA, NUM_IMAGES, IMG_MAX_SIZE
from facade_project.geometry.heatmap import points_to_cwh, build_heatmaps

# 000.torch because they are not rotated but only resized
IMG_RESIZED_TENSORS_PATH = ['{}/images/rot_aug_{}/img_{:03d}_000.torch'.format(PATH_TO_DATA, IMG_MAX_SIZE, i) \
                            for i in range(NUM_IMAGES)]
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


def __load_img_heatmaps(img_torch_path, heatmap_info):
    img = torch.load(img_torch_path)
    heatmaps = build_heatmaps(heatmap_info, max_size=max(img.shape[1:]))
    return img, heatmaps


def load_img_heatmaps(index, img_tensors_path=IMG_RESIZED_TENSORS_PATH, heatmap_infos=HEATMAP_INFOS):
    return __load_img_heatmaps(
        img_torch_path=img_tensors_path[index],
        heatmap_info=heatmap_infos[index],
    )
