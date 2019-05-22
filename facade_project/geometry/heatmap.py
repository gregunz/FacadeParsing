import math
import torch
from shapely.geometry import Polygon

from facade_project import \
    IMG_MAX_SIZE, \
    LABEL_NAME_TO_VALUE, \
    SIGMA_FIXED, \
    IS_SIGMA_FIXED, \
    SIGMA_SCALE, \
    HEATMAP_TYPES


def points_to_cwh(points):
    poly = Polygon(points)
    ctr = poly.centroid
    x, y = poly.envelope.exterior.coords.xy
    width = max([abs(x1 - x2) for x1, x2 in zip(x[:-1], x[1:])])
    height = max([abs(y1 - y2) for y1, y2 in zip(y[:-1], y[1:])])

    return round(ctr.x), round(ctr.y), round(width), round(height)


def crop_heatmap_info(info, bbox):
    info = info.copy()
    tl_x, tl_y, br_x, br_y = bbox

    info['img_width'] = br_x - tl_x
    info['img_height'] = br_y - tl_y
    for cwh in info['cwh_list']:
        c_x, c_y = cwh['center']
        cwh['center'] = (c_x - tl_x, c_y - tl_y)
    return info


def resize_heatmap_info(info, max_size):
    ratio = max_size / max(info['img_width'], info['img_height'])
    if ratio == 1:
        return info
    info = info.copy()
    info['img_width'] = round(ratio * info['img_width'])
    info['img_height'] = round(ratio * info['img_height'])
    for cwh in info['cwh_list']:
        cwh['width'] = round(ratio * cwh['width'])
        cwh['height'] = round(ratio * cwh['height'])
        cwh['center'] = tuple(round(ratio * c) for c in cwh['center'])
    return info


def build_heatmaps(
        heatmap_info,
        cropped_bbox=None,
        max_size=IMG_MAX_SIZE,
        label_name_to_value=LABEL_NAME_TO_VALUE,
        is_sigma_fixed=IS_SIGMA_FIXED,
        sigma_fixed=SIGMA_FIXED,
        sigma_scale=SIGMA_SCALE,
        heatmaps_types=HEATMAP_TYPES,
):
    handled_heatmap_types = ('center', 'width', 'height')

    n_heatmaps_per_class = len([h for h in heatmaps_types if h in handled_heatmap_types])
    assert n_heatmaps_per_class > 0, 'one must build at least one heatmap'

    if cropped_bbox is not None:
        heatmap_info = crop_heatmap_info(heatmap_info, cropped_bbox)
    if max_size is not None:
        heatmap_info = resize_heatmap_info(heatmap_info, max_size)

    img_height, img_width = heatmap_info['img_height'], heatmap_info['img_width']
    n_heatmaps = n_heatmaps_per_class * (len(label_name_to_value) - 1)  # no heatmap for the background class
    heatmaps = torch.zeros((n_heatmaps, img_height, img_width))

    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in [img_height, img_width]]
    )

    for cwh in heatmap_info['cwh_list']:
        heatmap_idx = label_name_to_value[cwh['label']] - 1
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

        heatmap_idx *= n_heatmaps_per_class
        for name in heatmaps_types:
            if name == 'center':
                heatmaps[heatmap_idx] += img_layer
                heatmap_idx += 1
            elif name == 'width':
                heatmaps[heatmap_idx] += img_layer * w
                heatmap_idx += 1
            elif name == 'height':
                heatmaps[heatmap_idx] += img_layer * h
                heatmap_idx += 1
            else:
                print('WARNING: unexpected heatmap type: {}'.format(name))

    return heatmaps
