
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