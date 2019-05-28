import PIL
import math
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor

from facade_project import IMG_MAX_SIZE
from facade_project.utils.misc import tf_if


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


def rescale(inputs, max_size=IMG_MAX_SIZE, itp_name='BI'):
    is_tensor = type(inputs) is torch.Tensor
    if is_tensor:
        h, w = inputs.shape[1:]
    else:
        w, h = inputs.size

    print(h, w, max_size)
    ratio = max_size / max(h, w)

    if ratio == 1:
        return inputs

    return T.Compose([
        tf_if(T.ToPILImage(), is_tensor),
        T.Resize((round(h * ratio), round(w * ratio)), interpolation=get_interpolation(itp_name)),
        tf_if(T.ToTensor(), is_tensor),
    ])(inputs)


def resize(inputs, size, itp_name='BI'):
    is_tensor = type(inputs) is torch.Tensor
    if is_tensor:
        h, w = inputs.shape[1:]
    else:
        w, h = inputs.size

    h2, w2 = size
    if h2 == h and w2 == w:
        return inputs

    return T.Compose([
        tf_if(T.ToPILImage(), is_tensor),
        T.Resize(size, interpolation=get_interpolation(itp_name)),
        tf_if(T.ToTensor(), is_tensor),
    ])(inputs)


def rotate(img, angle, itp_name='BI'):
    is_tensor = type(img) is Tensor
    img_to_new_dim = lambda img: rotated_rect_with_max_area(*img.size, angle * math.pi / 180)[::-1]
    return T.Compose([
        tf_if(T.ToPILImage(), is_tensor),
        T.Lambda(lambda img: TF.rotate(img, angle, resample=get_interpolation(itp_name))),
        T.Lambda(lambda img: T.CenterCrop(img_to_new_dim(img))(img)),
        tf_if(T.ToTensor(), is_tensor),
    ])(img)


def get_interpolation(name):
    if name == 'BI':
        return PIL.Image.BILINEAR
    elif name == 'NN':
        return PIL.Image.NEAREST
    else:
        assert False, 'interpolation not handled'
