import labelme
import matplotlib.pyplot as plt
import torch
from torchvision import utils

from facade_project import LABEL_NAME_TO_VALUE

DEFAULT_FIG_SIZE = (16, 9)


def convert_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy().transpose((1, 2, 0))
    return tensor


def show_img(image, fig_size=DEFAULT_FIG_SIZE):
    if isinstance(image, torch.Tensor):
        image = image.clone().permute(1, 2, 0)
    print(image.shape)
    plt.figure(figsize=fig_size)
    plt.imshow(image.squeeze())
    plt.show()


def show_labeled_img(image, label, label_names=LABEL_NAME_TO_VALUE, fig_size=DEFAULT_FIG_SIZE):
    """Show labeled image"""
    if isinstance(label_names, dict):
        label_names = {v: k for k, v in label_names.items()}
        label_names = [label_names[i] for i in label_names]

    image = convert_to_numpy(image)
    label = convert_to_numpy(label)
    img = (image * 255).astype('uint8')
    lbl = label.astype('uint8')
    if len(lbl.shape) == 3:
        lbl = lbl[:, :, 0]
    lbl_viz = labelme.utils.draw_label(lbl, img, label_names)
    show_img(lbl_viz, fig_size)


# Helper function to show a batch
def show_batch(images_batch, labels_batch, label_names, nrow=2, fig_size=DEFAULT_FIG_SIZE):
    """Show labeled image for a batch of samples."""
    img_grid = utils.make_grid(images_batch, nrow=nrow)
    img_grid = img_grid.numpy().transpose((1, 2, 0))

    lbl_grid = utils.make_grid(labels_batch, nrow=nrow)
    lbl_grid = lbl_grid.numpy()[0]

    # lbl_names = sorted(list({l for group in sample_batched['label_names'] for l in group}))

    show_labeled_img(img_grid, lbl_grid, label_names, fig_size=fig_size)
    plt.title('Batch from dataloader')


def show_channels(img, nrow=3, fig_size=DEFAULT_FIG_SIZE):
    channels = [img[i].unsqueeze(0) / img[i].max() for i in range(img.size(0))]
    img_grid = utils.make_grid(channels, nrow=nrow)
    show_img(img_grid, fig_size)
