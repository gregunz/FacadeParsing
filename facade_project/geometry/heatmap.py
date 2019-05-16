import math
import torch
from shapely.geometry import Polygon
from torchvision import transforms as T

from facade_project import IMG_MAX_SIZE, LABEL_NAME_TO_VALUE, SIGMA, SIGMA_FIXED, SIGMA_SCALE
from facade_project.data.augmentation import tf_if


def resize_heatmaps(inputs, targets=None):
    is_tensor = type(inputs) is torch.Tensor
    if is_tensor:
        h, w = inputs.shape[1:]
    else:
        w, h = inputs.size

    ratio = IMG_MAX_SIZE / max(h, w)
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


def label_me_to_heatmap(json_data, shape):
    # img = img_b64_to_arr(json_data['imageData'])
    height, width = shape[:2]

    n_heatmaps = 3 * (len(LABEL_NAME_TO_VALUE) - 1)
    heatmaps = torch.zeros((n_heatmaps, height, width))

    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in [height, width]]
    )

    for shape in json_data['shapes']:
        lbl = shape['label']
        if lbl in LABEL_NAME_TO_VALUE:
            points = shape['points']
            c_x, c_y, w, h = points_to_cwh(points)
            img_layer = 1
            for mean, std, mgrid in zip([c_y, c_x], [h, w], meshgrids):
                if SIGMA_FIXED:
                    std = SIGMA
                else:
                    std //= SIGMA_SCALE
                img_layer *= 1 / (std * math.sqrt(2 * math.pi)) * \
                             torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
            img_layer = img_layer / torch.max(img_layer)

            heatmap_idx = LABEL_NAME_TO_VALUE[lbl] - 1
            heatmap_idx *= 3

            heatmaps[heatmap_idx] += img_layer
            heatmaps[heatmap_idx + 1] += img_layer * w
            heatmaps[heatmap_idx + 2] += img_layer * h
    return heatmaps
