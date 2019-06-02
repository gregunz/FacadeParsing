import copy
import os
import sys
import time

import torch

from facade_project.utils.ml_utils import Epocher


def train_model(dataloaders, path_weights, model_name, model, device, criterion, optimizer, scheduler=None,
                metric_handler=None, writer=None, num_epoch=25, keep_n_best=3, verbose=True):
    since = time.time()

    model = model.to(device)

    epoch_offset = 0
    if hasattr(model, 'epoch_trained'):
        epoch_offset += model.epoch_trained

    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_paths = []
    best_loss = sys.maxsize

    epocher = Epocher(num_epoch, epoch_offset=epoch_offset)

    try:
        for epoch in epocher:
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    if scheduler:
                        scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                for data_idx, (inputs, targets) in enumerate(dataloaders[phase]):
                    epocher.print('{}: {}/{} batch'.format(phase, data_idx, len(dataloaders[phase])))

                    inputs = inputs.to(device)
                    if type(targets) is dict:
                        targets = {k: v.to(device) for k, v in targets.items()}
                    else:
                        targets = targets.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    loss_scalar = loss.item() * inputs.size(0)
                    running_loss += loss_scalar

                    # metric handler
                    if metric_handler:
                        metric_handler.add(outputs, targets)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)

                # metric handler
                is_best_metric = False
                if metric_handler:
                    is_best_metric = metric_handler.compute(phase=phase, dataset_size=len(dataloaders[phase]))

                if verbose:
                    metric_desc = ''
                    if metric_handler:
                        metric_desc = ' - {}'.format(metric_handler.description())
                    stats_string = '<{}> Loss: {:.4f}{}'.format(phase, epoch_loss, metric_desc)
                    epocher.update_stats(stats_string)

                # tensorboard
                if writer:
                    x_axis = epoch
                    writer.add_scalar('{}_loss'.format(phase), epoch_loss, x_axis)
                    if metric_handler:
                        for scalar_info in metric_handler.scalar_infos(phase):
                            writer.add_scalar(*scalar_info, x_axis)

                # deep copy the model
                if phase == 'val' and (epoch_loss < best_loss or is_best_metric):
                    best_loss = min(epoch_loss, best_loss)
                    model_path = '{}/{}_{:03d}.torch'.format(path_weights, model_name, epoch)
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
    if metric_handler:
        print(metric_handler.description_best())

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
