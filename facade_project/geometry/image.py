import math
import torch
import torchvision.transforms as T

from facade_project import IMG_MAX_SIZE
from facade_project.data.augmentation import tf_if


def rotated_rect_with_max_area(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) // cos_2a, (h * cos_a - w * sin_a) // cos_2a

    return wr, hr


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
    # TODO: NN interpolation for targets if it is a mask
    return resizer(inputs), resizer(targets)
