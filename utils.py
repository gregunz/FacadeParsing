import labelme
import json
import numpy as np
import math
import PIL
from torchvision.transforms import ToTensor

def to_multiple_of_shape(x, y, m, max_size):
    n_x = x // m
    n_y = y // m
    n_max_size = max_size // m
    
    if n_x > n_max_size and n_x > n_y:
        n_x = n_max_size
        n_y = round(y * n_max_size / x)
    else:
        n_y = n_max_size
        n_x = round(x * n_max_size / y)
        
    return (n_x * m, n_y * m)


def to_square_crops(x, y, size):
    assert x % size == 0
    assert y % size == 0
    crops = []
    for i in range(x//size):
        for j in range(y//size):
            crops.append((i * size, j * size))
    return crops

def path_to_tuple(img_path, label_name_to_value):
    data = json.load(open(img_path))

    imageData = data['imageData']
    img = labelme.utils.img_b64_to_arr(imageData)
    
    # removing object (and misnamed objet) class
    data['shapes'] = [shape for shape in data['shapes'] if shape['label'] in label_name_to_value]
        
    lbl = labelme.utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
    lbl = lbl.astype('uint8')[:, :, np.newaxis]
    
    return img, lbl

def load_png_tuple(img_dir, img_idx, rot_idx=None, as_tensor=False):
    if rot_idx is not None:
        rot_idx = '_{:03d}'.format(rot_idx)
    else:
        rot_idx = ''
    def load(s):
        return PIL.Image.open('{:s}/{:s}_{:03d}{:s}.png'.format(img_dir, s, img_idx, rot_idx))
    img = load('img')
    lbl = load('lbl')
    if as_tensor:
        img = ToTensor()(img)
        lbl = (ToTensor()(lbl) * 256).int()
    return img, lbl
    

def find_limits(lbl, step, margin):
    has_label = lambda bins: len(bins) > 1 or bins[0].item() != 0
    h, w = lbl.shape[1:]
    up, down, left, right = -1, -1, -1, -1
    for x1 in range(step, w, step):
        if has_label(lbl[:, :, x1-step:x1].unique()):
            left = max(0, x1 - (step + margin))
            break
    for x2 in range(step, w, step):
        if has_label(lbl[:, :, w-x2:w-x2+step].unique()):
            right = min(w, w - x2 + (step + margin))
            break
    for y1 in range(step, h, step):
        if has_label(lbl[:, y1-step:y1, :].unique()):
            up = max(0, y1 - (step + margin))
            break
    for y2 in range(step, h, step):
        if has_label(lbl[:, h-y2:h-y2+step, :].unique()):
            down = min(h, h - y2 + (step + margin))
            break
    assert up != -1 and down != -1 and left != -1 and right != -1
    return up, down, left, right

def rotated_rect_with_max_area(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)//cos_2a, (h*cos_a - w*sin_a)//cos_2a

    return wr,hr

