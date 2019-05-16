import json

import PIL
import labelme
import numpy as np
from labelme.utils import img_b64_to_arr
from torchvision.transforms import ToTensor

from facade_project import LABEL_NAME_TO_VALUE
from facade_project.geometry.heatmap import points_to_cwh


def load_tuple_from_json(img_path, label_name_to_value=LABEL_NAME_TO_VALUE):
    data = json.load(open(img_path))

    image_data = data['imageData']
    img = labelme.utils.img_b64_to_arr(image_data)

    # removing object (and misnamed objet) class
    data['shapes'] = [shape for shape in data['shapes'] if shape['label'] in label_name_to_value]

    lbl = labelme.utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
    lbl = lbl.astype('uint8')[:, :, np.newaxis]

    return img, lbl


def load_tuple_from_png(img_dir, img_idx, rot_idx=None, as_tensor=False):
    if rot_idx is not None:
        rot_idx = '_{:03d}'.format(rot_idx)
    else:
        rot_idx = ''

    def load(s):
        return PIL.Image.open('{:s}/{:s}_{:03d}{:s}.png'.format(img_dir, s, img_idx, rot_idx))

    img = load('img')
    lbl = load('lbl')
    if as_tensor:
        img = ToTensor()(img)
        lbl = (ToTensor()(lbl) * 256).int()
    return img, lbl


def load_heatmaps_info(labelme_json_path):
    json_data = json.load(open(labelme_json_path, mode='r'))
    return extract_heatmaps_info(json_data)


def extract_heatmaps_info(json_data):
    img = img_b64_to_arr(json_data['imageData'])
    info = {
        'img_height': img.shape[0],
        'img_width': img.shape[1],
        'cwh_list': [],
    }
    for shape in json_data['shapes']:
        lbl = shape['label']
        if lbl in LABEL_NAME_TO_VALUE:
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
