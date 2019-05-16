import torch
from shapely.geometry import Polygon
from torchvision import transforms as T

from facade_project import IMG_MAX_SIZE
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
