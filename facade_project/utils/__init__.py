import torch

from facade_project.geometry.heatmap import heatmaps_to_info, info_to_mask


def outputs_to_predictions(outputs, center_threshold=20, surface_threshold=1000):
    """
    Going from the outputs of the model (predicting heatmaps) to the final predictions

    :param outputs: torch.Tensor, outputs from the model
    :param center_threshold: float, the threshold used to extract heatmaps centers
    :param surface_threshold: float, the minimum surface area of an extracted object
    :return: tuple(torch.Tensor, torch.Tensor), two masks, one from the segmentation, one from the regression
    """
    assert outputs.size(0) == 7, '1 channel for each of the 4 labels, 3 for the heatmaps'

    logits = outputs[:4]
    heatmaps = torch.relu(outputs[4:])

    info = heatmaps_to_info(heatmaps, logits, center_threshold, surface_threshold)
    heatmaps_mask = info_to_mask(info)

    mask = logits.max(0)[1].unsqueeze(0)
    return mask, heatmaps_mask
