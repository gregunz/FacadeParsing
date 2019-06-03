import torch
from torch.nn import functional as F


def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
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
    """
    # assert logits.shape == true.shape
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


def facade_criterion(predictions_list, predictions_weights, num_classes, use_dice=True):
    def facade_criterion_closure(outputs, targets):
        assert len(predictions_list) == len(predictions_weights)

        losses = []
        output_idx = 0
        for p in predictions_list:
            target = targets[p]

            if p == 'mask':
                output = outputs[:, output_idx:output_idx + num_classes]
                output_idx += num_classes

                if use_dice:
                    losses.append(dice_loss(output, target))
                else:
                    losses.append(F.cross_entropy(output, target))

            elif p == 'center' or p == 'width' or p == 'height':
                n_channels = target.size(1)
                output = outputs[:, output_idx:output_idx + n_channels]
                output_idx += n_channels
                losses.append(F.mse_loss(output, F.relu(target)))

        assert output_idx == outputs.size(1), 'we used all the channels available for the loss'

        loss = losses[0]
        if len(losses) > 1:
            for l in losses[1:]:
                loss = loss + l

        return loss

    return facade_criterion_closure
