
## previous dataset _init_
def remove_ext(s):
    return os.path.splitext(s)[0] 

if only_labeled:
    images_names = {remove_ext(s) for s in os.listdir(img_dir)}
    labels_names = {remove_ext(s) for s in os.listdir(labels_dir)}
    names_with_labels = images_names.intersection(labels_names)
    self.img_paths = [os.path.join(img_dir, s) for s in sorted(os.listdir(img_dir)) if remove_ext(s) in names_with_labels]
    self.labels_paths = [os.path.join(labels_dir, s) for s in sorted(os.listdir(labels_dir)) if remove_ext(s) in names_with_labels]
else:
    self.img_paths = [os.path.join(img_dir, s) for s in sorted(os.listdir(img_dir))]
    self.labels_paths = [os.path.join(labels_dir, s) for s in sorted(os.listdir(labels_dir))]

self.transform = transform


## collate_fn
from torch.utils.data.dataloader import default_collate

def my_collate(batch):
    lbl_names = dict()
    for i, b in enumerate(batch):
        lbl_names[i] = b['label_names']
        del b['label_names']
    batch_collated_no_names = default_collate(batch)
    batch_collated_no_names['label_names'] = lbl_names
    return batch_collated_no_names



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        sample = sample.copy()
        sample['image'] = torch.from_numpy(image)
        sample['label'] = torch.from_numpy(label)
        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        sample = sample.copy()
        sample['image'] = image
        sample['label'] = label
        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = skimage.transform.resize(image, (new_h, new_w))
        lbl = skimage.transform.resize(label, (new_h, new_w), preserve_range=True).astype('uint8')

        sample = sample.copy()
        sample['image'] = img
        sample['label'] = lbl
        return sample
