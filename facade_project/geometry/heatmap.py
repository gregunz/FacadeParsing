import copy

import math
import torch
from shapely.geometry import Polygon

from facade_project import \
    SIGMA_FIXED, \
    IS_SIGMA_FIXED, \
    SIGMA_SCALE, \
    HEATMAP_TYPES_HANDLED
from facade_project.geometry.image import rotated_rect_with_max_area


def points_to_cwh(points):
    poly = Polygon(points)
    ctr = poly.centroid
    x, y = poly.envelope.exterior.coords.xy
    width = max([abs(x1 - x2) for x1, x2 in zip(x[:-1], x[1:])])
    height = max([abs(y1 - y2) for y1, y2 in zip(y[:-1], y[1:])])

    return round(ctr.x), round(ctr.y), round(width), round(height)


def crop(info, bbox):
    info = copy.deepcopy(info)
    tl_x, tl_y, br_x, br_y = bbox

    info['img_width'] = br_x - tl_x
    info['img_height'] = br_y - tl_y
    for cwh in info['cwh_list']:
        c_x, c_y = cwh['center']
        cwh['center'] = (c_x - tl_x, c_y - tl_y)
    return info


def rescale(info, max_size):
    ratio = max_size / max(info['img_width'], info['img_height'])
    return rescale_with_ratios(info, ratio, ratio)


def rescale_with_ratios(info, width_ratio, height_ratio):
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
    if type(size) is int:
        size = (size, size)
    height_ratio = size[0] / info['img_height']
    width_ratio = size[1] / info['img_width']
    return rescale_with_ratios(info, width_ratio=width_ratio, height_ratio=height_ratio)


def rotate(info, angle):
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
