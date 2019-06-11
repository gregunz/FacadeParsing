import torch

from facade_project import HEATMAP_LABELS
from facade_project.geometry.heatmap import heatmaps_to_info
from facade_project.utils.ml_utils import MetricHandler


def jaccard_index(pred, target, n_classes):
    """
    Compute jaccard index (intersection over union) metric

    :param pred: torch.Tensor, prediction mask
    :param target: torch.Tensor, target mask
    :param n_classes: int, number of classes
    :return: torch.Tensor, jaccard index per class
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    assert pred.shape == target.shape, '{} != {}'.format(pred.shape, target.shape)

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


def accuracy_from_infos(pred_info, target_info, heatmap_labels=HEATMAP_LABELS, ret_details=False):
    """
    Computing the accuracy of a predicted heatmaps info compared to the target heatmaps info

    :param pred_info: dict, predicted heatmaps info
    :param target_info: dict, target heatmaps info
    :param heatmap_labels: tuple, labels of interest
    :param ret_details: bool, whether to return more detailed informations
    :return: torch.Tensor or tuple(torch.Tensor, torch.Tensor) if ret_details is True
    """
    img_width = target_info['img_width']
    img_height = target_info['img_height']

    accuracies = []

    def to_ranges(info, label):
        ranges = []
        for cwh in info['cwh_list']:
            cwh_label = cwh['label']
            if cwh_label in heatmap_labels and (label is None or cwh_label == label):
                c_x, c_y = cwh['center']
                w, h = cwh['width'], cwh['height']

                from_y = int(round(max(0, c_y - h // 2)))
                to_y = int(round(min(img_height, c_y + h // 2)))
                from_x = int(round(max(0, c_x - w // 2 + 1)))
                to_x = int(round(min(img_width, c_x + w // 2)))
                ranges.append((from_x, to_x, from_y, to_y, cwh_label))
        return ranges

    def is_correct(cwh, ranges, any_label=False):
        c_x, c_y = cwh['center']
        for from_x, to_x, from_y, to_y, target_label in ranges:
            if (any_label or cwh['label'] == target_label) and from_x <= c_x <= to_x and from_y <= c_y <= to_y:
                return True
        return False

    if ret_details:
        details = torch.zeros(3, len(heatmap_labels))

    for idx, label in enumerate(heatmap_labels):
        target_ranges = to_ranges(target_info, label)
        pred_ranges = to_ranges(pred_info, label)

        true_pos = [is_correct(cwh, pred_ranges) for cwh in target_info['cwh_list']]

        if len(target_ranges) == 0:
            accuracies.append(float('nan'))
        else:
            accuracies.append(sum(true_pos) / len(target_ranges))

        if ret_details:
            details[idx][0] = sum(true_pos)
            details[idx][1] = len(target_ranges)

    if ret_details:
        # accuracy of center (without correct label)
        details[-1][0] = sum(
            [is_correct(cwh, to_ranges(pred_info, None), any_label=True) for cwh in target_info['cwh_list']])
        details[-1][1] = len(to_ranges(target_info, None))
        return torch.tensor(accuracies), details

    return torch.tensor(accuracies)


class FacadeMetric(MetricHandler):
    """
    A Metric Handler for the facade parsing task.

    As of 2019-06-05, it handles jaccard index during traning for better logging
    """

    def __init__(self, predictions_list, label_name_to_value, heatmap_labels):
        super().__init__()
        self.predictions_list = predictions_list
        self.label_name_to_value = label_name_to_value
        self.n_classes = len(label_name_to_value)
        self.heatmap_labels = heatmap_labels

        if self.do_mask():
            self.jacc_best_mean = torch.zeros(self.n_classes)
            self.jacc_run = []
            self.jacc_epochs = {'train': [], 'val': []}
        if self.do_heatmaps():
            self.heatmaps_acc_best = torch.zeros(len(self.heatmap_labels))
            self.heatmaps_acc_run = []
            self.heatmaps_acc_epochs = {'train': [], 'val': []}

    def do_mask(self):
        return 'mask' in self.predictions_list

    def do_heatmaps(self):
        # disable because computationally too expensive
        return False

    def add(self, outputs, targets):
        assert type(targets) is dict

        for p in self.predictions_list:
            assert p in targets

        if self.do_mask():
            _, preds = torch.max(outputs[:, :4], 1)

            self.jacc_run.append(jaccard_index(preds, targets['mask'], self.n_classes))

        if self.do_heatmaps():
            logits = outputs[:, :4]
            for idx, heatmaps_info in enumerate(targets['heatmaps_info']):
                heatmaps = outputs[idx, 4:]
                pred_info = heatmaps_to_info(heatmaps.detach().cpu(), logits[idx], 10, 1000,
                                             heatmap_labels=self.heatmap_labels)
                acc = accuracy_from_infos(pred_info, heatmaps_info.info)
                self.heatmaps_acc_run.append(acc)

    def compute(self, phase, dataset_size):
        assert phase in ['train', 'val']

        if self.do_mask():
            jacc_run_per_class = torch.stack(self.jacc_run).unbind(dim=1)
            jacc_epoch = torch.tensor([j[torch.isnan(j) == 0].mean() for j in jacc_run_per_class])
            self.jacc_epochs[phase].append(jacc_epoch)
            # best jaccard is always on validation data
            if phase == 'val' and self.last_is_best(phase):
                self.jacc_best_mean = jacc_epoch
            # init running jaccard index
            self.jacc_run = []

        if self.do_heatmaps():
            heatmaps_acc_per_class = torch.stack(self.heatmaps_acc_run).unbind(dim=1)
            heatmaps_acc_epoch = torch.tensor([h[torch.isnan(h) == 0].mean() for h in heatmaps_acc_per_class])
            self.heatmaps_acc_epochs[phase].append(heatmaps_acc_epoch)
            if phase == 'val' and self.last_is_best(phase):
                self.heatmaps_acc_best = heatmaps_acc_epoch
            self.heatmaps_acc_run = []

    def scalar_infos(self, phase):
        infos = []
        if self.do_mask():
            for name, value in self.label_name_to_value.items():
                scalar_name = 'jacc_{}_{}'.format(name, phase)
                scalar = self.jacc_epochs[phase][-1][value].item()
                infos.append((scalar_name, scalar))

        if self.do_heatmaps():
            for idx, name in enumerate(self.heatmap_labels):
                scalar_name = 'heatmap_acc_{}_{}'.format(name, phase)
                scalar = self.heatmaps_acc_epochs[phase][-1][idx].item()
                infos.append((scalar_name, scalar))

        return infos

    def last_is_best(self, phase):

        if self.do_mask():
            jacc_means = [j.mean().item() for j in self.jacc_epochs[phase]]
            if len(jacc_means) == 1:
                return True
            if jacc_means[-1] > max(jacc_means[:-1]):
                return True

        if self.do_heatmaps():
            heatmaps_acc_means = [acc.mean().item() for acc in self.heatmaps_acc_epochs[phase]]
            if len(heatmaps_acc_means) == 1:
                return True
            if heatmaps_acc_means[-1] > max(heatmaps_acc_means[:-1]):
                return True

        return False

    def description(self, phase):
        s = ''
        if self.do_mask():
            s += 'jacc epoch = [{:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}]'.format(
                *self.jacc_epochs[phase][-1].tolist(),
            )
        return s

    def description_best(self):
        s = ''
        if self.do_mask():
            s += 'best jacc epoch mean = [{:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}]'.format(
                *self.jacc_best_mean.tolist(),
            )
        return s
