import torch
from torch.nn import functional as F

from facade_project import FACADE_ROT_PROPORTIONS


def dice_loss(logits, true, eps=1e-7):
    """
    Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        true: a tensor of shape [B, 1, H, W].
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.

    credits to : https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return 1 - dice_loss


def facade_criterion(predictions_list, predictions_weights, device, num_classes, use_dice=True, center_factor=90.):
    """
    Criterion for facade parsing.

    Handle 'mask' segmentation and 'heatmaps' regression

    :param predictions_list: list(str), which predictions to minimize (mask and/or heatmaps)
    :param predictions_weights: list(float), weights associated with the loss of single predictions
    :param num_classes: int, number of classes of the mask
    :param use_dice: bool, whether to use dice loss or cross entropy loss
    :param center_factor: float, a factor multiplied to the center heatmap target which would otherwise be too small
    in comparision to width and height.
    :return: function, the criterion to use for training
    """
    assert len(predictions_list) > 0
    assert len(predictions_list) == len(predictions_weights)
    def facade_criterion_closure(outputs, targets):

        losses = []
        output_idx = 0

        for p in predictions_list:
            target = targets[p]
            n_channels = target.size(1)

            if p == 'mask':
                assert n_channels == 1, 'target is a one-channel mask'
                output = outputs[:, output_idx:output_idx + num_classes]
                output_idx += num_classes

                if use_dice:
                    losses.append(dice_loss(output, target))
                else:
                    percentages = torch.tensor(FACADE_ROT_PROPORTIONS, device=device)
                    assert num_classes == len(percentages)
                    inv_perc = 1 / percentages
                    weights = inv_perc / inv_perc.sum()
                    losses.append(F.cross_entropy(output, target.squeeze(1), weight=weights))

            elif p == 'heatmaps':
                if n_channels == 3:
                    # this means, there is the center which needs to be scaled
                    target[:, 0] = target[:, 0] * center_factor
                else:
                    assert 1 <= n_channels <= 2, 'only handling center, width and height maps'

                output = F.relu(outputs[:, output_idx:output_idx + n_channels])
                output_idx += n_channels
                losses.append(F.mse_loss(output, target))

        assert output_idx == outputs.size(1), 'we used all the channels available for the loss'

        loss = torch.zeros(1, device=device)
        for l, w in zip(losses, predictions_weights):
            loss = loss + l * w
        return loss

    return facade_criterion_closure
