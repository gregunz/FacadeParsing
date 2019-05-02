import random
import math

import torch
import PIL
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torch import Tensor
from PIL import Image, ImageChops

from utils import find_limits, rotated_rect_with_max_area
from constants import crop_margin, crop_step, patch_size

def get_bbox(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff)
    bbox = diff.getbbox()
    return bbox
    
def trim(im, bbox):
    if bbox:
        w,h = im.size
        bbox = max(0, bbox[0] - crop_margin),\
            max(0, bbox[1] - crop_margin),\
            min(w, bbox[2] + crop_margin),\
            min(h, bbox[3] + crop_margin)
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
        #T.Lambda(lambda img: img if not is_label else (img * 255).int()),
    ])(img)

def gen_2_rot(img, lbl):
    assert type(img) is type(lbl)
    
    neg_angle = random.randint(-10, -5)
    pos_angle = random.randint(5, 10)
    return [(img, lbl),\
             (random_rot(img, neg_angle, False), random_rot(lbl, neg_angle, True)),\
             (random_rot(img, pos_angle, False), random_rot(lbl, pos_angle, True))]

def cut_borders(img, lbl):
    if type(img) is Tensor:
        up, down, left, right = find_limits(lbl, crop_step, crop_margin)
        return img[:, up:down, left:right], lbl[:, up:down, left:right]
    else:
        bbox = get_bbox(lbl)
        return trim(img, bbox), trim(lbl, bbox)
    
def random_crop_and_resize(img, crop_size, resize_size, is_label):
    is_tensor = type(img) is Tensor
    
    itp = 0 if is_label else 2 #0 is nearest, 2 is bilinear interpolation
    
    return T.Compose([
        tf_if(T.ToPILImage(), is_tensor),
        T.RandomCrop(crop_size),
        tf_if(T.Resize(resize_size, interpolation=itp), crop_size != resize_size),
        tf_if(T.ToTensor(), is_tensor),
    ])(img)

def gen_crops(img, lbl, crop_size, ncrops):
    assert type(img) is type(lbl)
    
    ls = []
    
    seed = random.random()
    for i in range(ncrops):
        random.seed(seed + i)
        img_crop = random_crop_and_resize(img, crop_size, patch_size, is_label=False)
        random.seed(seed + i)
        lbl_crop = random_crop_and_resize(lbl, crop_size, patch_size, is_label=True)
        ls.append((img_crop, lbl_crop))
    return ls

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
    
    # TODO inconsistencies: changes on tensor are not exactly the same
    # as the one with PIL (scaling of factors)
    if is_tensor:
        #apply contrast
        img = contr_factor * (img - 0.5) + 0.5
        #apply brightness
        img = img + (bright_factor - 1)
        return torch.clamp(img , min=0, max=1), lbl        
    else:
        img = PIL.ImageEnhance.Contrast(img).enhance(contr_factor)
        img = PIL.ImageEnhance.Brightness(img).enhance(bright_factor)
        return img, lbl


def gen_one_patch(full_res_img, full_res_lbl):
    is_tensor = type(full_res_img) is Tensor
    assert type(full_res_img) is type(full_res_lbl)
    
    if is_tensor:
        full_res_img, full_res_lbl = tuple_to_pil(full_res_img, full_res_lbl)
    
    img, lbl = full_res_img, full_res_lbl
    angle = random.randint(-10, 10)
    img, lbl = random_rot(img, angle), random_rot(lbl, angle, is_label=True)
    img, lbl = cut_borders(img, lbl)
    
    crop_size = min(images_labels[0][0].size) // 3
    img, lbl = gen_crops(img, lbl, crop_size=crop_size, ncrops=ncrops)[0]
    img, lbl = random_flip(img, lbl)
    img, lbl = random_brightness_and_contrast(img, lbl)
    return img, lbl

def gen_one_rotation(full_res_img, full_res_lbl, as_tensor=True):
    is_tensor = type(full_res_img) is Tensor
    assert type(full_res_img) is type(full_res_lbl)
    
    if is_tensor:
        full_res_img, full_res_lbl = tuple_to_pil(full_res_img, full_res_lbl)
    
    img, lbl = full_res_img, full_res_lbl
    angle = random.randint(-10, 10)
    img, lbl = random_rot(img, angle, is_label=False), random_rot(lbl, angle, is_label=True)
    img, lbl = cut_borders(img, lbl)
    
    w,h = img.size
    resize_factor = min(1, 3 * patch_size / min(w,h))
    if resize_factor < 1:
        def resizer(is_label):
            itp = 0 if is_label else 2 #0 is nearest, 2 is bilinear interpolation
            return T.Resize((round(h*resize_factor),round(w*resize_factor)), interpolation=itp)
        img = resizer(False)(img)
        lbl = resizer(True)(lbl)
    
    if as_tensor:
        img, lbl = T.ToTensor()(img), (T.ToTensor()(lbl)* 255).int()
        
    return img, lbl

def gen_one_patch(full_res_img, full_res_lbl, as_tensor=True):
    is_tensor = type(full_res_img) is Tensor
    assert type(full_res_img) is type(full_res_lbl)
    
    if is_tensor:
        full_res_img, full_res_lbl = tuple_to_pil(full_res_img, full_res_lbl)
    
    img, lbl = full_res_img, full_res_lbl
    angle = random.randint(-10, 10)
    img, lbl = random_rot(img, angle, is_label=False), random_rot(lbl, angle, is_label=True)
    img, lbl = cut_borders(img, lbl)
    
    crop_size = min(img.size) // 3
    img, lbl = gen_crops(img, lbl, crop_size=crop_size, ncrops=1)[0]
    img, lbl = random_flip(img, lbl)
    img, lbl = random_brightness_and_contrast(img, lbl)
    
    if as_tensor:
        img, lbl = T.ToTensor()(img), (T.ToTensor()(lbl)* 255).int()
        
    return img, lbl

def to_stack_tensors(images_labels):
    img_patches = torch.stack([T.ToTensor()(img) for (img, _) in images_labels])
    lbl_patches = torch.stack([T.ToTensor()(lbl) for (_, lbl) in images_labels])

    return img_patches, (lbl_patches * 255).int()

def aug_rotated(rotated_img, rotated_lbl, ncrops, crop_size=patch_size, as_tensor=True):
    images_labels = gen_crops(rotated_img, rotated_lbl, crop_size=crop_size, ncrops=ncrops)
    images_labels = [random_flip(img, lbl) for (img, lbl) in images_labels]
    images_labels = [random_brightness_and_contrast(img, lbl) for (img, lbl) in images_labels]
    
    if as_tensor:
        return to_stack_tensors(images_labels)
    else:
        return images_labels

def aug_pipeline(full_res_img, full_res_lbl, ncrops_multiplier=1, as_tensor=True):
    """
    From full res images to patches.
    This is done randomly set the seed to get predictable results
    """
    is_tensor = type(full_res_img) is Tensor
    assert type(full_res_img) is type(full_res_lbl)
    
    if is_tensor:
        full_res_img, full_res_lbl = tuple_to_pil(full_res_img, full_res_lbl)
        
    images_labels = gen_2_rot(full_res_img, full_res_lbl)
    images_labels = [cut_borders(img, lbl) for (img, lbl) in images_labels]
    
    w,h = images_labels[0][0].size
    crop_size = min(w, h) // 3
    ncrops = round(0.5 * ncrops_multiplier * w/crop_size * h/crop_size)
    
    images_labels = [crop for img,lbl in images_labels for crop in aug_rotated(img,lbl, ncrops=ncrops, crop_size=crop_size, as_tensor=False)]
    
    if as_tensor:
        return to_stack_tensors(images_labels)
    else:
        return images_labels

