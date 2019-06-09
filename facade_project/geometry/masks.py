from torch import Tensor

from facade_project import CUT_MARGIN


def find_limits(mask, step, margin):
    """
    Find iteratively the limits where the mask starts having non zero values

    :param mask: torch.Tensor
    :param step: int, number of rows/columns in which we check there are non zeros values in one iteration
    :param margin: margin to add on the limits at the end
    :return: torch.Tensor, cropped mask
    """
    h, w = mask.shape[1:]
    up, down, left, right = -1, -1, -1, -1
    for x1 in range(step, w, step):
        if mask[:, :, x1 - step:x1].sum().item() > 0:
            left = max(0, x1 - (step + margin))
            break
    for x2 in range(step, w, step):
        if mask[:, :, w - x2:w - x2 + step].sum().item() > 0:
            right = min(w, w - x2 + (step + margin))
            break
    for y1 in range(step, h, step):
        if mask[:, y1 - step:y1, :].sum().item() > 0:
            up = max(0, y1 - (step + margin))
            break
    for y2 in range(step, h, step):
        if mask[:, h - y2:h - y2 + step, :].sum().item() > 0:
            down = min(h, h - y2 + (step + margin))
            break
    assert up != -1 and down != -1 and left != -1 and right != -1
    return up, down, left, right


def get_bbox(im, cut_margin=CUT_MARGIN):
    """
    Compute the exact bounding where the mask is non zero

    :param im: PIL.Image, image
    :param cut_margin: int, extending the bounding box by a small margin
    :return: tuple(int, int, int, int), bounding box (top left x, y, bottom right x, y)
    """
    bbox = im.getbbox()

    w, h = im.size
    bbox = max(0, bbox[0] - cut_margin), \
           max(0, bbox[1] - cut_margin), \
           min(w, bbox[2] + cut_margin), \
           min(h, bbox[3] + cut_margin)

    return bbox


def crop_pil(im, bbox):
    """
    Crop an image given a bounding box

    :param im: PIL.Image
    :param bbox: bounding box
    :return: PIL.Image, cropped image
    """
    if bbox:
        return im.crop(bbox)
    return im


def cut_borders(img, lbl):
    """
    Crop an image and its labels

    :param img: torch.Tensor or PIL.Image
    :param lbl: torch.Tensor or PIL.Image
    :return: tuple(torch.Tensor or PIL.Image, torch.Tensor or PIL.Image)
    """
    if type(img) is Tensor:
        up, down, left, right = find_limits(lbl, step=10, margin=CUT_MARGIN)
        return img[:, up:down, left:right], lbl[:, up:down, left:right]
    else:
        bbox = get_bbox(lbl, cut_margin=CUT_MARGIN)
        return crop_pil(img, bbox), crop_pil(lbl, bbox)
