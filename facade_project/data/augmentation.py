import random

import PIL
import torch
import torchvision.transforms.functional as TF
from torch import Tensor

from facade_project.geometry.heatmap import crop as crop_info, flip as flip_info, HeatmapsInfo
from facade_project.geometry.image import rotate


def random_rot(angle_limits, itp_name='BI'):
    """
    return function which apply random rotation within angle limits (in degrees) to an image and its label
    :param angle_limits: tuple(from, to) (integer only)
    :param itp_name: interpolation used ('BI' for bilinear and 'NN' for nearest neighbor)
    :return: a function
    """

    def random_rot_closure(img, lbl):
        angle = random.randint(*angle_limits)
        tf = lambda inputs: rotate(inputs, angle, itp_name=itp_name)
        tf = handle_dict(tf)
        return tf(img), tf(lbl)

    return random_rot_closure


def random_crop(crop_size):
    """
    return a function which apply random crop to an image and its label

    :param crop_size: int or tuple(height, width)
    :return: a function
    """

    if type(crop_size) is int:
        crop_x, crop_y = crop_size, crop_size
    else:
        crop_x, crop_y = crop_size

    def random_crop_closure(img, lbl):
        assert type(img) is torch.Tensor

        h, w = img.shape[1:]
        top = random.randint(0, h - crop_y)
        left = random.randint(0, w - crop_x)

        def tf(inputs):
            if type(inputs) is HeatmapsInfo:
                inputs = HeatmapsInfo(crop_info(inputs.info, (left, top, left + crop_x, top + crop_y)))
            else:
                inputs = inputs[:, top:top + crop_y, left:left + crop_x]
            return inputs

        tf = handle_dict(tf)
        return tf(img), tf(lbl)

    return random_crop_closure


def random_flip(p=0.5):
    """
    return a function which apply random flip to an image and its label given a probability

    :param p: probability to flip
    :return: a function
    """

    def random_flip_closure(img, lbl):
        is_tensor = type(img) is Tensor
        rand = random.random()

        def tf(inputs):
            if rand < p:
                if type(inputs) is HeatmapsInfo:
                    return HeatmapsInfo(flip_info(inputs.info))
                elif is_tensor:
                    return inputs.flip(2)
                else:
                    return TF.hflip(inputs)
            return inputs

        tf = handle_dict(tf)
        return tf(img), tf(lbl)

    return random_flip_closure


def random_brightness_and_contrast(contrast_from=0.8, brightness_from=0.9):
    """
    return a function which apply random change of brightness to an image

    :param contrast_from: the smaller, the more contrast will be potentially added
    :param brightness_from: the smaller, the more brightness will be potentially added
    :return: a function
    """

    def random_brightness_and_contrast_closure(img):
        is_tensor = type(img) is Tensor
        contr_factor = contrast_from + random.random() * (2 - contrast_from * 2)
        bright_factor = brightness_from + random.random() * (2 - brightness_from * 2)

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
    """
    return a function which will apply sequentially the transforms function to an image and its label

    :param transforms: list of transform functions
    :return: a function
    """

    def sequential_apply(img, lbl):
        for tf in transforms:
            img, lbl = tf(img, lbl)
        return img, lbl

    return sequential_apply


def handle_dict(tf):
    """
    return a transform function which handles dict as inputs by applying it to dict values

    :param tf: a function
    :return: the same function handling dict
    """

    def tf_handling_dict(inputs):
        if type(inputs) is dict:
            outputs = dict()
            for k, v in inputs.items():
                outputs[k] = tf(v)
            return outputs
        return tf(inputs)

    return tf_handling_dict
