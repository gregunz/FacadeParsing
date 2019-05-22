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


def build_heatmaps(
        heatmap_info,
        max_size=IMG_MAX_SIZE,
        label_name_to_value=LABEL_NAME_TO_VALUE,
        is_sigma_fixed=IS_SIGMA_FIXED,
        sigma_fixed=SIGMA_FIXED,
        sigma_scale=SIGMA_SCALE,
        heatmaps_types=HEATMAP_TYPES,
):
    img_height, img_width = heatmap_info['img_height'], heatmap_info['img_width']
    ratio = 1
    if max_size is not None:
        ratio = max_size / max(img_height, img_width)

    img_height = round(ratio * img_height)
    img_width = round(ratio * img_width)
    n_heatmaps_per_class = len([h for h in heatmaps_types if h in HEATMAP_TYPES])
    assert n_heatmaps_per_class > 0, 'one must build at least one heatmap'

    n_heatmaps = n_heatmaps_per_class * (len(label_name_to_value) - 1)  # no heatmap for the background class
    heatmaps = torch.zeros((n_heatmaps, img_height, img_width))

    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in [img_height, img_width]]
    )

    for cwh in heatmap_info['cwh_list']:
        heatmap_idx = label_name_to_value[cwh['label']] - 1
        center, w, h = cwh['center'], cwh['width'], cwh['height']
        center = [round(c * ratio) for c in center][::-1]
        w, h = round(ratio * w), round(ratio * h)

        img_layer = 1
        for mean, std, mgrid in zip(center, [h, w], meshgrids):
            if is_sigma_fixed:
                std = sigma_fixed
                std = round(ratio * std)
            else:
                std *= sigma_scale
            img_layer *= torch.exp(-((mgrid - mean) / (2 * std)) ** 2) / (std * math.sqrt(2 * math.pi))
        img_layer = img_layer / torch.max(img_layer)

        heatmap_idx *= 3
        for name in heatmaps_types:
            if name == HEATMAP_TYPES[0]:
                heatmaps[heatmap_idx] += img_layer
            if name == HEATMAP_TYPES[1]:
                heatmaps[heatmap_idx + 1] += img_layer * w
            if name == HEATMAP_TYPES[2]:
                heatmaps[heatmap_idx + 2] += img_layer * h

    return heatmaps
