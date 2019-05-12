import labelme
import json
import numpy as np
import PIL

from torchvision.transforms import ToTensor


def load_tuple_from_json(img_path, label_name_to_value):
    data = json.load(open(img_path))

    image_data = data['imageData']
    img = labelme.utils.img_b64_to_arr(image_data)

    # removing object (and misnamed objet) class
    data['shapes'] = [shape for shape in data['shapes'] if shape['label'] in label_name_to_value]

    lbl = labelme.utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
    lbl = lbl.astype('uint8')[:, :, np.newaxis]

    return img, lbl


def load_tuple_from_png(img_dir, img_idx, rot_idx=None, as_tensor=False):
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


