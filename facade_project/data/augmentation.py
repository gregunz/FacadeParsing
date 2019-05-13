import random

import PIL
import math
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageChops
from torch import Tensor

from facade_project import CUT_STEP, CUT_MARGIN
from facade_project.utils.geometry import rotated_rect_with_max_area, find_limits


def get_bbox(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff)
    bbox = diff.getbbox()
    return bbox


def trim(im, bbox):
    if bbox:
        w, h = im.size
        bbox = max(0, bbox[0] - CUT_MARGIN), \
               max(0, bbox[1] - CUT_MARGIN), \
               min(w, bbox[2] + CUT_MARGIN), \
               min(h, bbox[3] + CUT_MARGIN)
        return im.crop(bbox)
    return im


def tf_if(tf, do_tf=False):
    return T.Lambda(lambda img: tf(img) if do_tf else img)


def tuple_to_pil(img, lbl):
    return T.ToPILImage()(img), T.ToPILImage()(lbl)


def random_rot(img, angle, is_label):
    is_tensor = type(img) is Tensor

    img_to_new_dim = lambda img: rotated_rect_with_max_area(*img.size, angle * math.pi / 180)[::-1]
    return T.Compose([
        tf_if(T.ToPILImage(), is_tensor),
        T.Lambda(lambda img: TF.rotate(img, angle)),
        T.Lambda(lambda img: T.CenterCrop(img_to_new_dim(img))(img)),
        tf_if(T.ToTensor(), is_tensor),
        # T.Lambda(lambda img: img if not is_label else (img * 255).int()),
    ])(img)


def cut_borders(img, lbl):
    if type(img) is Tensor:
        up, down, left, right = find_limits(lbl, CUT_STEP, CUT_MARGIN)
        return img[:, up:down, left:right], lbl[:, up:down, left:right]
    else:
        bbox = get_bbox(lbl)
        return trim(img, bbox), trim(lbl, bbox)


def random_crop_and_resize(img, crop_size, resize_size, is_label):
    is_tensor = type(img) is Tensor

    itp = 0 if is_label else 2  # 0 is nearest, 2 is bilinear interpolation

    return T.Compose([
        tf_if(T.ToPILImage(), is_tensor),
        T.RandomCrop(crop_size),
        tf_if(T.Resize(resize_size, interpolation=itp), crop_size != resize_size),
        tf_if(T.ToTensor(), is_tensor),
    ])(img)


def random_flip(img, lbl, p=0.5):
    is_tensor = type(img) is Tensor
    assert type(img) is type(lbl)

    if random.random() < p:
        tf = T.Compose([
            tf_if(T.ToPILImage(), is_tensor),
            T.Lambda(lambda img: TF.hflip(img)),
            tf_if(T.ToTensor(), is_tensor),
        ])
        return tf(img), tf(lbl)
    else:
        return img, lbl


def random_brightness_and_contrast(img, lbl):
    is_tensor = type(img) is Tensor
    assert type(img) is type(lbl)

    contr_from = 0.75
    contr_factor = contr_from + random.random() * (2 - contr_from * 2)
    bright_from = 0.85
    bright_factor = bright_from + random.random() * (2 - bright_from * 2)

    # TODO inconsistencies: changes on tensor are not exactly the same as the one with PIL (scaling of factors)
    if is_tensor:
        # apply contrast
        img = contr_factor * (img - 0.5) + 0.5
        # apply brightness
        img = img + (bright_factor - 1)
        return torch.clamp(img, min=0, max=1), lbl
    else:
        img = PIL.ImageEnhance.Contrast(img).enhance(contr_factor)
        img = PIL.ImageEnhance.Brightness(img).enhance(bright_factor)
        return img, lbl
