import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor

from facade_project.geometry.image import rotate
from facade_project.utils.misc import tf_if


def tuple_to_pil(img, lbl):
    return T.ToPILImage()(img), T.ToPILImage()(lbl)


def random_rot(img, angle_limits, itp_name='BI'):
    angle = random.randint(*angle_limits)
    return rotate(img, angle, itp_name=itp_name)


def random_crop(img, lbl, crop_size):
    assert type(img) is torch.Tensor

    if type(crop_size) is int:
        crop_x, crop_y = crop_size, crop_size
    else:
        crop_x, crop_y = crop_size

    h, w = img.shape[1:]
    top = random.randint(0, h - crop_y)
    left = random.randint(0, w - crop_x)
    img = img[:, top:top + crop_y, left:left + crop_x]
    lbl = lbl[:, top:top + crop_y, left:left + crop_x]

    img = random_brightness_and_contrast(img)
    return img, lbl


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


def random_brightness_and_contrast(img):
    is_tensor = type(img) is Tensor

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
        return torch.clamp(img, min=0, max=1)
    else:
        img = PIL.ImageEnhance.Contrast(img).enhance(contr_factor)
        img = PIL.ImageEnhance.Brightness(img).enhance(bright_factor)
        return img


def compose(transforms):
    def sequential_apply(img, lbl):
        for tf in transforms:
            img, lbl = tf(img, lbl)
        return img, lbl

    return sequential_apply
