import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor

from facade_project.geometry.image import rotate


def tuple_to_pil(img, lbl):
    return T.ToPILImage()(img), T.ToPILImage()(lbl)


def random_rot(angle_limits, itp_name='BI'):
    def random_rot_closure(img, lbl):
        angle = random.randint(*angle_limits)
        tf = lambda inputs: rotate(inputs, angle, itp_name=itp_name)
        tf = handle_dict(tf)
        return tf(img), tf(lbl)

    return random_rot_closure


def random_crop(crop_size):
    def tf(inputs):
        assert type(inputs) is torch.Tensor

        if type(crop_size) is int:
            crop_x, crop_y = crop_size, crop_size
        else:
            crop_x, crop_y = crop_size

        h, w = inputs.shape[1:]
        top = random.randint(0, h - crop_y)
        left = random.randint(0, w - crop_x)
        inputs = inputs[:, top:top + crop_y, left:left + crop_x]

        return inputs

    tf = handle_dict(tf)

    def random_crop_closure(img, lbl):
        return tf(img), tf(lbl)

    return random_crop_closure


def random_flip(p=0.5):
    def random_flip_closure(img, lbl):
        is_tensor = type(img) is Tensor

        if random.random() < p:
            if is_tensor:
                tf = lambda x: x  # TODO handle tensor flip
            else:
                tf = TF.hflip

            tf = handle_dict(tf)
            return tf(img), tf(lbl)
        else:
            return img, lbl

    return random_flip_closure


def random_brightness_and_contrast(contrast_from=0.75, brightness_from=0.85):
    def random_brightness_and_contrast_closure(img):
        is_tensor = type(img) is Tensor
        contr_factor = contrast_from + random.random() * (2 - contrast_from * 2)
        bright_factor = brightness_from + random.random() * (2 - brightness_from * 2)

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

    return random_brightness_and_contrast_closure


def compose(transforms):
    def sequential_apply(img, lbl):
        for tf in transforms:
            img, lbl = tf(img, lbl)
        return img, lbl

    return sequential_apply


def handle_dict(tf):
    def tf_handling_dict(inputs):
        if type(inputs) is dict:
            return {k: tf(v) for k, v in inputs.items()}
        return tf(inputs)

    return tf_handling_dict
