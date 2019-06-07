import copy

import math
import torch
from labelme.utils import img_b64_to_arr
from scipy import ndimage
from shapely.geometry import Polygon

from facade_project import SIGMA_FIXED, IS_SIGMA_FIXED, SIGMA_SCALE, HEATMAP_TYPES_HANDLED, LABEL_NAME_TO_VALUE, \
    HEATMAP_LABELS
from facade_project.geometry.image import rotated_rect_with_max_area


def points_to_cwh(points):
    """
    Given list of points (coordinates) representing a polygon, compute the envelope of it and
    return its center coordinates, width and height
    :param points: list(tuple(int, int))
    :return: tuple(center_x, center_y, width, height)
    """
    poly = Polygon(points)
    ctr = poly.centroid
    x, y = poly.envelope.exterior.coords.xy
    width = max([abs(x1 - x2) for x1, x2 in zip(x[:-1], x[1:])])
    height = max([abs(y1 - y2) for y1, y2 in zip(y[:-1], y[1:])])

    return round(ctr.x), round(ctr.y), round(width), round(height)


def flip(info):
    """
    Flip horizontally a heatmap info

    :param info: dict, heatmaps info
    :return: dict, heatmaps info (copy)
    """
    info = copy.deepcopy(info)

    for cwh in info['cwh_list']:
        c_x, c_y = cwh['center']
        cwh['center'] = (info['img_width'] - c_x, c_y)

    return info


def crop(info, bbox):
    """
    Crop a heatmap given its info and a cropping area

    :param info: dict, heatmaps info
    :param bbox: tuple(int, int, int, int), bounding box (top left x, top left y, bottom right x, bottom right, y) of the cropping area
    :return: dict, heatmaps info (copy)
    """
    info = copy.deepcopy(info)
    tl_x, tl_y, br_x, br_y = bbox

    info['img_width'] = br_x - tl_x
    info['img_height'] = br_y - tl_y
    for cwh in info['cwh_list']:
        c_x, c_y = cwh['center']
        cwh['center'] = (c_x - tl_x, c_y - tl_y)

    info['cwh_list'] = [cwh for cwh in info['cwh_list'] \
                        if 0 <= cwh['center'][0] <= info['img_width'] and 0 <= cwh['center'][1] <= info['img_height']]

    return info


def rescale(info, max_size):
    """
    Rescale a heatmap given its info and a maximum size.
    Biggest side (width or height) will be rescaled to maximum size, and the smallest
    will be rescaled proportionally

    :param info: dict, heatmaps info
    :param max_size: int
    :return: dict, heatmaps info rescaled (copy)
    """
    ratio = max_size / max(info['img_width'], info['img_height'])
    return rescale_with_ratios(info, ratio, ratio)


def rescale_with_ratios(info, width_ratio, height_ratio):
    """
    Rescale heatmaps sides given its info and ratios.

    :param info: dict, heatmaps info
    :param width_ratio: float, ratio by which width will be multiplied
    :param height_ratio: float, ratio by which height will be multiplied
    :return: dict, heatmaps info rescaled (copy)
    """
    info = copy.deepcopy(info)

    if width_ratio == 1 and height_ratio == 1:
        return info

    info['img_width'] = round(width_ratio * info['img_width'])
    info['img_height'] = round(height_ratio * info['img_height'])
    for cwh in info['cwh_list']:
        cwh['width'] = round(width_ratio * cwh['width'])
        cwh['height'] = round(height_ratio * cwh['height'])
        cwh['center'] = (cwh['center'][0] * width_ratio, cwh['center'][1] * height_ratio)
    return info


def resize(info, size):
    """
    Resize heatmaps given its info and a target size

    :param info: dict, heatmaps info
    :param size: int or tuple(int, int)
    :return: dict, heatmaps infos resized (copy)
    """
    if type(size) is int:
        size = (size, size)
    height_ratio = size[0] / info['img_height']
    width_ratio = size[1] / info['img_width']
    return rescale_with_ratios(info, width_ratio=width_ratio, height_ratio=height_ratio)


def rotate(info, angle):
    """
    Rotated heatmaps given its info and an angle

    :param info: dict, heatmaps info
    :param angle: int, angle in degrees
    :return: dict, heatmaps infos rotated (copy)
    """
    info = copy.deepcopy(info)
    if angle == 0:
        return info

    angle = angle * math.pi / 180
    center_x, center_y = info['img_width'] / 2, info['img_height'] / 2
    sin_a, cos_a = math.sin(angle), math.cos(angle)

    def rotate_coordinate(x, y):
        x -= center_x
        y -= center_y
        x_rot = cos_a * x + sin_a * y
        y_rot = cos_a * y - sin_a * x
        return round(x_rot + center_x), round(y_rot + center_y)

    # rotate centers
    for cwh in info['cwh_list']:
        cwh['center'] = rotate_coordinate(*cwh['center'])

    # crop
    wr, hr = rotated_rect_with_max_area(info['img_width'], info['img_height'], angle)
    cropped_bbox = center_x - (wr // 2 + wr % 2), center_y - (hr // 2 + hr % 2), center_x + wr // 2, center_y + hr // 2
    cropped_bbox = tuple(round(e) for e in cropped_bbox)
    info = crop(info, cropped_bbox)
    return info


def build_heatmaps(
        heatmap_info,
        labels,
        heatmap_types,
        is_sigma_fixed=IS_SIGMA_FIXED,
        sigma_fixed=SIGMA_FIXED,
        sigma_scale=SIGMA_SCALE,
):
    """
    Build a tensor of the heatmaps given its info

    :param heatmap_info: dict
    :param labels: list(str), which labels to put on the heatmaps (e.g. door and window)
    :param heatmap_types: list(str), which heatmap to build (center, width, height)
    :param is_sigma_fixed: bool, whether the sigma of the gaussian is fixed for all points
    :param sigma_fixed: float, value of the fixed sigma
    :param sigma_scale: float, if sigma is not fixed, width and height (for x and y) will
     be used and rescaled by this value
    :return: torch.Tensor
    """
    heatmap_types_to_idx = {htype: idx for idx, htype in enumerate(HEATMAP_TYPES_HANDLED) if htype in heatmap_types}
    assert len(heatmap_types_to_idx) > 0, 'one must build at least one heatmap'

    img_height, img_width = heatmap_info['img_height'], heatmap_info['img_width']
    heatmaps = torch.zeros(len(heatmap_types_to_idx), img_height, img_width)

    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in [img_height, img_width]]
    )

    for cwh in heatmap_info['cwh_list']:
        if cwh['label'] in labels:
            center = cwh['center'][::-1]  # x and y instead of y and x
            h, w = cwh['height'], cwh['width']
            img_layer = 1
            for mean, std, mgrid in zip(center, [h, w], meshgrids):
                if is_sigma_fixed:
                    std = sigma_fixed  # overriding std value
                else:
                    std *= sigma_scale
                img_layer *= torch.exp(-((mgrid - mean) / (2 * std)) ** 2) / (std * math.sqrt(2 * math.pi))
            img_layer = img_layer / torch.max(img_layer)

            for name, heatmap_idx in heatmap_types_to_idx.items():
                if name == 'center':
                    heatmaps[heatmap_idx] += img_layer
                elif name == 'width':
                    heatmaps[heatmap_idx] += img_layer * w
                elif name == 'height':
                    heatmaps[heatmap_idx] += img_layer * h
                else:
                    print('WARNING: unexpected heatmap type: {}'.format(name))

    return heatmaps


def extract_heatmaps_info(json_data, label_name_to_value=LABEL_NAME_TO_VALUE):
    """
    Given labelme json data, construct a heatmaps info

    :param json_data: dict, labelme data from a json file
    :param label_name_to_value: dict
    :return: dict, heatmaps info
    """
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


def heatmaps_to_info(heatmaps, logits, center_threshold, surface_threshold, label_to_name_value=LABEL_NAME_TO_VALUE,
                     heatmap_labels=HEATMAP_LABELS):
    """
    Extracting back infos from heatmaps (center, width, height)

    :param heatmaps: torch.Tensor, 3 x H x W tensor (3 because of center, width, height)
    :param logits: torch.Tensor, logits output from the model to define the true label
    :param center_threshold: float, the threshold used to extract heatmaps centers
    :param surface_threshold: float, the minimum surface area of an extracted object
    :param label_to_name_value: dict, label name mapped to its value
    :param heatmap_labels: tuple, label to extract
    :return: dict, heatmaps info
    """
    heatmap_center = heatmaps[0].numpy()
    img_height, img_width = heatmap_center.shape
    mask, nb = ndimage.label(heatmap_center > center_threshold)

    cwh_list = []
    for label in range(nb):
        centers = ndimage.measurements.center_of_mass((mask == label).astype(float) * heatmap_center)
        c_y, c_x = tuple(int(round(c)) for c in centers)

        w = heatmaps[1].numpy()[mask == label].max()
        h = heatmaps[2].numpy()[mask == label].max()

        if w * h > surface_threshold:

            from_y = int(round(max(0, c_y - h // 2)))
            to_y = int(round(min(img_height, c_y + h // 2)))

            from_x = int(round(max(0, c_x - w // 2)))
            to_x = int(round(min(img_width, c_x + w // 2)))

            label = 'window'
            if logits is not None:
                max_num_pixels = 0
                for l in heatmap_labels:
                    v = label_to_name_value[l]
                    num_pixels = (logits[:, from_y:to_y, from_x:to_x].max(0)[1] == v).sum()
                    if num_pixels > max_num_pixels:
                        max_num_pixels = num_pixels
                        label = l

            cwh = {
                'center': (c_x, c_y),
                'width': w,
                'height': h,
                'label': label,  # default for now, but could also be door
            }
            cwh_list.append(cwh)
    return {
        'img_width': img_width,
        'img_height': img_height,
        'cwh_list': cwh_list,
    }


def info_to_mask(info, label_to_name_value=LABEL_NAME_TO_VALUE, heatmap_labels=HEATMAP_LABELS):
    """
    Creating a mask given heatmaps info

    :param info: dict, heatmaps info
    :param label_to_name_value: dict, label name mapped to its value
    :param heatmap_labels: tuple, label to extract
    :return: torch.Tensor
    """
    img_width = info['img_width']
    img_height = info['img_height']
    mask = torch.zeros((img_height, img_width), dtype=torch.int)
    for cwh in info['cwh_list']:
        if cwh['label'] in heatmap_labels:
            c_x, c_y = cwh['center']
            w, h = cwh['width'], cwh['height']

            from_y = int(round(max(0, c_y - h // 2)))
            to_y = int(round(min(img_height, c_y + h // 2)))

            from_x = int(round(max(0, c_x - w // 2 + 1)))
            to_x = int(round(min(img_width, c_x + w // 2)))

            label = label_to_name_value[cwh['label']]
            mask[from_y:to_y, from_x:to_x] = label
    return mask.unsqueeze(0)


def is_heatmaps_info(info):
    return type(info) is dict and \
           type(info.get('img_width')) is int and \
           type(info.get('img_height')) is int and \
           type(info.get('cwh_list')) is list


class HeatmapsInfo:
    def __init__(self, info):
        assert is_heatmaps_info(info)
        self.info = info
