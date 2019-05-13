import copy
import os
import sys
import time

import torch

from facade_project import LABEL_NAME_TO_VALUE
from facade_project.utils.metrics import jaccard_index
from facade_project.utils.tqdm_ml import Epocher


def train_model(dataloaders, dataset_sizes, path_to_data, model_name, model, device, criterion, optimizer, scheduler,
                writer=None, num_epoch=25, keep_n_best=3, verbose=True):
    since = time.time()

    epoch_offset = 1  # because we start at 1 and not 0
    if hasattr(model, 'epoch_trained'):
        epoch_offset += model.epoch_trained

    best_model_wts = copy.deepcopy(model.state_dict())
    best_jacc_mean = 0.0
    best_loss = sys.maxsize

    best_model_paths = []

    epocher = Epocher(num_epoch, epoch_offset=epoch_offset)

    try:
        for epoch in epocher:
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_jacc = torch.tensor([0.0] * model.num_classes)

                # Iterate over data.
                for data_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                    epocher.print('{}: {}/{} batch'.format(phase, data_idx, len(dataloaders[phase])))
                    inputs = inputs.to(device)
                    labels = labels.to(device).squeeze(1)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    loss_scalar = loss.item() * inputs.size(0)
                    running_loss += loss_scalar
                    jacc_scalars = jaccard_index(preds, labels, model.num_classes)
                    running_jacc += jacc_scalars

                    # tensorboard
                    if writer:
                        x_axis = data_idx + epoch * dataset_sizes[phase]
                        writer.add_scalar('{}_loss'.format(phase), loss_scalar, x_axis)
                        for name, value in LABEL_NAME_TO_VALUE.items():
                            writer.add_scalar('{}_jacc_{}'.format(phase, name), jacc_scalars[value], x_axis)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_jacc = running_jacc / dataset_sizes[phase]
                epoch_jacc_mean = epoch_jacc.mean().item()

                if verbose:
                    jacc_list = ['{}: {:.4f}'.format(n, epoch_jacc[v].item()) for n, v in LABEL_NAME_TO_VALUE.items()]
                    stats_string = '<{}> Loss: {:.4f} - MeanJacc: {:.4f} - JaccPerClass: ({})'.format(phase, epoch_loss,
                                                                                                      epoch_jacc_mean,
                                                                                                      ' '.join(
                                                                                                          jacc_list))
                    epocher.update_stats(stats_string)

                # deep copy the model
                if phase == 'val' and (epoch_loss < best_loss or epoch_jacc_mean > best_jacc_mean):
                    best_jacc_mean = max(epoch_jacc_mean, best_jacc_mean)
                    best_loss = min(epoch_loss, best_loss)
                    model_path = '{}/models/model_{}_{}.torch'.format(path_to_data, model_name, epoch)
                    if len(best_model_paths) >= keep_n_best:
                        os.remove(best_model_paths.pop(0))
                    best_model_paths.append(model_path)

                    torch.save(model.state_dict(), model_path)
                    if verbose:
                        ls_string = model_path
                        epocher.update_ls(ls_string)
                    best_model_wts = copy.deepcopy(model.state_dict())

            if hasattr(model, 'epoch_trained'):
                model.epoch_trained += 1

    except KeyboardInterrupt:
        pass

    time_elapsed = time.time() - since
    print()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    print('Best val Jacc: {:4f}'.format(best_jacc_mean))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
