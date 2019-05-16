import math
import torch
from shapely.geometry import Polygon
from torchvision import transforms as T

from facade_project import IMG_MAX_SIZE, LABEL_NAME_TO_VALUE, SIGMA, SIGMA_FIXED, SIGMA_SCALE
from facade_project.data.augmentation import tf_if


def resize(inputs, targets=None, max_size=IMG_MAX_SIZE):
    is_tensor = type(inputs) is torch.Tensor
    if is_tensor:
        h, w = inputs.shape[1:]
    else:
        w, h = inputs.size

    ratio = max_size / max(h, w)
    resizer = T.Compose([
        tf_if(T.ToPILImage(), is_tensor),
        tf_if(T.Resize((round(h * ratio), round(w * ratio))), ratio != 1),
        tf_if(T.ToTensor(), is_tensor),
    ])

    if targets is None:
        return resizer(inputs)
    return resizer(inputs), resizer(targets)


def points_to_cwh(points):
    poly = Polygon(points)
    ctr = poly.centroid
    x, y = poly.envelope.exterior.coords.xy
    width = max([abs(x1 - x2) for x1, x2 in zip(x[:-1], x[1:])])
    height = max([abs(y1 - y2) for y1, y2 in zip(y[:-1], y[1:])])

    return round(ctr.x), round(ctr.y), round(width), round(height)


def build_heatmaps(heatmap_info, max_size=None, label_name_to_value=LABEL_NAME_TO_VALUE):
    img_height, img_width = heatmap_info['img_height'], heatmap_info['img_width']
    ratio = 1
    if max_size is not None:
        ratio = max_size / max(img_height, img_width)

    img_height = round(ratio * img_height)
    img_width = round(ratio * img_width)
    n_heatmaps = 3 * (len(label_name_to_value) - 1)  # no heatmap for the background class
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
            # TODO: don't use constants inside the function directly
            if SIGMA_FIXED:
                std = SIGMA
                std = round(ratio * std)
            else:
                std //= SIGMA_SCALE
            img_layer *= torch.exp(-((mgrid - mean) / (2 * std)) ** 2) / (std * math.sqrt(2 * math.pi))
        img_layer = img_layer / torch.max(img_layer)

        heatmap_idx *= 3
        heatmaps[heatmap_idx] += img_layer
        heatmaps[heatmap_idx + 1] += img_layer * w
        heatmaps[heatmap_idx + 2] += img_layer * h
    return heatmaps
