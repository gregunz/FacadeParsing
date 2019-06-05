import torch

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


class FacadeMetric(MetricHandler):
    """
    A Metric Handler for the facade parsing task.

    As of 2019-06-05, it handles jaccard index during traning for better logging
    """
    def __init__(self, predictions_list, label_name_to_value):
        super().__init__()
        self.predictions_list = predictions_list
        self.label_name_to_value = label_name_to_value
        self.n_classes = len(label_name_to_value)
        self.metric_dict = dict()

        for p in self.predictions_list:
            if p == 'mask':
                self.metric_dict['jacc_best_mean'] = torch.zeros(self.n_classes)
                self.metric_dict['jacc_run'] = []
                self.metric_dict['jacc_epoch_train'] = []
                self.metric_dict['jacc_epoch_val'] = []
            elif p == 'center':
                pass
            elif p == 'width':
                pass
            elif p == 'height':
                pass

    def add(self, outputs, targets):
        assert type(targets) is dict
        for p in self.predictions_list:
            assert p in targets

        for p in self.predictions_list:
            if p == 'mask':
                _, preds = torch.max(outputs[:, :self.n_classes], 1)
                self.metric_dict['jacc_run'].append(jaccard_index(preds, targets['mask'], self.n_classes))
            elif p == 'center':
                pass
            elif p == 'width':
                pass
            elif p == 'height':
                pass

    def compute(self, phase, dataset_size):
        assert phase in ['train', 'val']

        for p in self.predictions_list:
            if p == 'mask':
                jacc_run_per_class = torch.stack(self.metric_dict['jacc_run']).unbind(dim=1)
                jacc_epoch = torch.tensor([j[torch.isnan(j) == 0].mean() for j in jacc_run_per_class])
                self.metric_dict['jacc_epoch_{}'.format(phase)].append(jacc_epoch)
                # best jaccard is always on validation data
                if phase == 'val' and self.last_is_best(phase):
                    self.metric_dict['jacc_best_mean'] = jacc_epoch
                # init running jaccard index
                self.metric_dict['jacc_run'] = []
            elif p == 'center':
                pass
            elif p == 'width':
                pass
            elif p == 'height':
                pass

    def scalar_infos(self, phase):
        infos = []
        for p in self.predictions_list:
            if p == 'mask':
                for name, value in self.label_name_to_value.items():
                    scalar_name = 'jacc_{}_{}'.format(name, phase)
                    scalar = self.metric_dict['jacc_epoch_{}'.format(phase)][-1][value].item()
                    infos.append((scalar_name, scalar))
            elif p == 'center':
                pass
            elif p == 'width':
                pass
            elif p == 'height':
                pass
        return infos

    def last_is_best(self, phase):
        jacc_epochs = self.metric_dict['jacc_epoch_{}'.format(phase)]
        jacc_epochs = [j.mean().item() for j in jacc_epochs]
        if len(jacc_epochs) == 1:
            return True
        return jacc_epochs[-1] > max(jacc_epochs[:-1])

    def description(self, phase):
        return 'jacc_epoch_{} = [{:4f}, {:4f}, {:4f}, {:4f}]'.format(
            phase,
            *self.metric_dict['jacc_epoch_{}'.format(phase)][-1].tolist(),
        )

    def description_best(self):
        return 'jacc_best_mean = [{:4f}, {:4f}, {:4f}, {:4f}]'.format(
            *self.metric_dict['jacc_best_mean'].tolist(),
        )
