import json

import labelme
import numpy as np

from facade_project import LABEL_NAME_TO_VALUE
from facade_project.geometry.heatmap import extract_heatmaps_info


def load_tuple_from_json(img_path, label_name_to_value=LABEL_NAME_TO_VALUE):
    """
    Load image and its associated mask from a labelme style json file

    :param img_path: str, path to the json file
    :param label_name_to_value: dict, which labels to extract and its value on the mask
    :return: tuple(numpy.ndarray, numpy.ndarray)
    """
    data = json.load(open(img_path))

    image_data = data['imageData']
    img = labelme.utils.img_b64_to_arr(image_data)

    data['shapes'] = [shape for shape in data['shapes'] if shape['label'] in label_name_to_value]

    lbl = labelme.utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
    lbl = lbl.astype('uint8')[:, :, np.newaxis]

    return img, lbl


def load_heatmaps_info(labelme_json_path):
    """
    Load heatmaps infos given labelme style json file

    :param labelme_json_path: str, path to the json file
    :return: dict, heatmaps info
    """
    json_data = json.load(open(labelme_json_path, mode='r'))
    return extract_heatmaps_info(json_data)
