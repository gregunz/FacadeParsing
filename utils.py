import labelme
import json
import numpy as np

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
    
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name == 'objet':
            shape['label'] = 'object'
            label_name = shape['label']
        label_value = label_name_to_value[label_name]
        
    lbl = labelme.utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
    lbl = lbl.astype('uint8')[:, :, np.newaxis]
    
    return img, lbl

def find_limits(lbl, margin):
    has_label = lambda bins: len(bins) > 1 or bins[0].item() != 0
    h, w = lbl.shape[1:]
    up, down, left, right = -1, -1, -1, -1
    for x1 in range(1, w, margin):
        if has_label(lbl[:, :, :x1].unique()):
            left = max(0, x1 - margin)
            break
    for x2 in range(1, w, margin):
        if has_label(lbl[:, :, -x2:].unique()):
            right = min(w, w - x2 + margin)
            break
    for y1 in range(1, h, margin):
        if has_label(lbl[:, :y1, :].unique()):
            up = max(0, y1 - margin)
            break
    for y2 in range(1, h, margin):
        if has_label(lbl[:, -y2:, :].unique()):
            down = min(h, h - y2 + margin)
            break
    assert up != -1 and down != -1 and left != -1 and right != -1
    return up, down, left, right

