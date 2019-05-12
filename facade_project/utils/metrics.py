import torch


def jaccard_index(pred, target, n_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    # for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    for cls in range(n_classes):  # This goes from 0:n_classes-1 -> class "0" is NOT ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return torch.tensor(ious)
