import PIL
import math
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor

from facade_project import IMG_MAX_SIZE


def rotated_rect_with_max_area(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.

    :param w: int, width
    :param h: int, height
    :param angle: float, angle in radians
    :return: tuple(int, int), width and height
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

    return round(wr), round(hr)


def rescale(inputs, max_size=IMG_MAX_SIZE, itp_name='BI'):
    """
    Rescale an image given a maximum size.

    Biggest side (width or height) will be rescaled to maximum size, and the smallest
    will be rescaled proportionally

    :param inputs: PIL or Tensor, image
    :param max_size: int
    :param itp_name: interpolation used ('BI' for bilinear and 'NN' for nearest neighbor)
    :return: PIL or Tensor, rescaled image
    """
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
        __tf_if__(T.ToPILImage(), is_tensor),
        T.Resize((round(h * ratio), round(w * ratio)), interpolation=__get_interpolation__(itp_name)),
        __tf_if__(T.ToTensor(), is_tensor),
    ])(inputs)


def resize(inputs, size, itp_name='BI'):
    """
    Resize  an image given a target size

    :param inputs: PIL or Tensor, image
    :param size: tuple(int, int)
    :param itp_name: interpolation used ('BI' for bilinear and 'NN' for nearest neighbor)
    :return: PIL or Tensor, resized image
    """
    is_tensor = type(inputs) is torch.Tensor
    if is_tensor:
        h, w = inputs.shape[1:]
    else:
        w, h = inputs.size

    h2, w2 = size
    if h2 == h and w2 == w:
        return inputs

    return T.Compose([
        __tf_if__(T.ToPILImage(), is_tensor),
        T.Resize(size, interpolation=__get_interpolation__(itp_name)),
        __tf_if__(T.ToTensor(), is_tensor),
    ])(inputs)


def rotate(img, angle, itp_name='BI'):
    """
    Rotate an image given an angle in degrees

    :param img: PIL or Tensor, image
    :param angle: int, angle of the rotation in degrees
    :param itp_name: interpolation used ('BI' for bilinear and 'NN' for nearest neighbor)
    :return: PIL or Tensor, rotated image
    """
    is_tensor = type(img) is Tensor
    img_to_new_dim = lambda img: rotated_rect_with_max_area(*img.size, angle * math.pi / 180)[::-1]
    return T.Compose([
        __tf_if__(T.ToPILImage(), is_tensor),
        T.Lambda(lambda img: TF.rotate(img, angle, resample=__get_interpolation__(itp_name), expand=False)),
        T.Lambda(lambda img: T.CenterCrop(img_to_new_dim(img))(img)),
        __tf_if__(T.ToTensor(), is_tensor),
    ])(img)


def __get_interpolation__(name):
    if name == 'BI':
        return PIL.Image.BILINEAR
    elif name == 'NN':
        return PIL.Image.NEAREST
    else:
        assert False, 'interpolation not handled'


def __tf_if__(tf, do_tf=False):
    return T.Lambda(lambda img: tf(img) if do_tf else img)
