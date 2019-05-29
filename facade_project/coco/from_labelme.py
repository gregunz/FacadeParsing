import json

import PIL.Image
import imageio
import numpy as np
from labelme import utils
from tqdm.auto import tqdm


class Labelme2coco(object):
    def __init__(self, labelme_json=None, save_json_path='./new.json', only_labels=None, save_img_dir=None):
        if labelme_json is None:
            labelme_json = []
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.only_labels = only_labels
        self.save_img_dir = save_img_dir
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

    def data_transfer(self):
        for num, json_file in enumerate(tqdm(self.labelme_json)):
            with open(json_file, 'r') as fp:
                data = json.load(fp)
                self.images.append(self.image(data, num))
                for shapes in data['shapes']:
                    label = shapes['label']
                    if (self.only_labels is None or label in self.only_labels) and label not in self.label:
                        self.categories.append(self.category(label))
                        self.label.append(label)
                    if label in self.label:
                        points = shapes['points']
                        if len(points) > 2:
                            self.annotations.append(self.annotation(points, label, num))
                            self.annID += 1

    def image(self, data, num):
        image = {}
        img = utils.img_b64_to_arr(data['imageData'])
        height, width = img.shape[:2]
        image['height'] = height
        image['width'] = width
        image['id'] = num
        image['file_name'] = data['imagePath'].split('/')[-1]
        self.height = height
        self.width = width
        if self.save_img_dir:
            imageio.imwrite(self.save_img_dir + '/' + image['file_name'], img)
        return image

    def category(self, label):
        category = {
            'supercategory': 'facade',
            'name': label
        }
        if self.only_labels is not None:
            category['id'] = self.only_labels[label]
        else:
            category['id'] = len(self.label) + 1
        return category

    def annotation(self, points, label, num):
        annotation = {
            'segmentation': [[p for point in points for p in point]],
            'iscrowd': 0,
            'image_id': num,
            'bbox': list(map(float, self.getbbox(points))),
            'category_id': self.get_cat_id(label),
            'id': self.annID
        }
        return annotation

    def get_cat_id(self, label):
        for category in self.categories:
            if label == category['name']:
                return category['id']
        return -1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [left_top_c, left_top_r, right_bottom_c - left_top_c, right_bottom_r - left_top_r]

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)
