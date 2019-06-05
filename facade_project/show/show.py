import labelme
import matplotlib.pyplot as plt
import torch
from torchvision import utils

from facade_project import LABEL_NAME_TO_VALUE

DEFAULT_FIG_SIZE = (16, 9)


def __convert_to_numpy__(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy().transpose((1, 2, 0))
    return tensor


def show_img(image, fig_size=DEFAULT_FIG_SIZE):
    """
    Plot an image

    :param image: torch.Tensor or numpy.ndarray
    :param fig_size: tuple(int, int), size of the plot
    :return: None
    """
    if isinstance(image, torch.Tensor):
        image = image.clone().permute(1, 2, 0)
    print(image.shape)
    plt.figure(figsize=fig_size)
    plt.imshow(image.squeeze())
    plt.show()


def show_labeled_img(image, label, label_names=LABEL_NAME_TO_VALUE, fig_size=DEFAULT_FIG_SIZE):
    """
    Plot an image with superposed label

    :param image: torch.Tensor or numpy.ndarray or numpy.ndarray
    :param label: torch.Tensor or numpy.ndarray
    :param label_names: list or dict, name of the labels
    :param fig_size: tuple(int, int), size of the plot
    :return: None
    """
    if isinstance(label_names, dict):
        label_names = {v: k for k, v in label_names.items()}
        label_names = [label_names[i] for i, _ in enumerate(label_names)]

    image = __convert_to_numpy__(image)
    label = __convert_to_numpy__(label)
    img = (image * 255).astype('uint8')
    lbl = label.astype('uint8')
    if len(lbl.shape) == 3:
        lbl = lbl[:, :, 0]
    lbl_viz = labelme.utils.draw_label(lbl, img, label_names)
    show_img(lbl_viz, fig_size)


def show_channels(img, nrow=3, fig_size=DEFAULT_FIG_SIZE):
    """
    Plot the channel of an image next to each others.

    :param img: torch.Tensor
    :param nrow: number of image per row
    :param fig_size: tuple(int, int), size of the plot
    :return: None
    """
    channels = [img[i].unsqueeze(0) / img[i].max() for i in range(img.size(0))]
    img_grid = utils.make_grid(channels, nrow=nrow)
    show_img(img_grid, fig_size)
