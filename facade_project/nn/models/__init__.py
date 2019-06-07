import json

import torch

from facade_project import PATH_TO_DATA
from facade_project.nn.models.albunet import AlbuNet
from facade_project.nn.models.unet import UNet


def load_trained_model(name, epoch, device):
    dir_path = '{}/models/{}'.format(PATH_TO_DATA, name)
    weights_path = '{}/weights_{:03d}.torch'.format(dir_path, epoch)
    summary = json.load(open('{}/summary.json'.format(dir_path), mode='r'))

    model_name = summary['model']
    num_outputs = 0
    for p in summary['predictions']:
        if p == 'mask':
            num_outputs += 4
        elif p == 'heatmaps':
            num_outputs += 3
    wf = 4
    if 'wf' in summary:
        wf = summary['wf']

    if model_name == 'albunet':
        model = AlbuNet(
            num_classes=num_outputs,
            num_filters=2 ** wf,
            pretrained=False,
            is_deconv=True,
        )
    else:
        model = UNet(
            in_channels=3,
            n_classes=num_outputs,
            wf=wf,
            padding=True,
            batch_norm=True,
            up_mode='upconv'
        )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    if hasattr(model, 'epoch_trained'):
        model.epoch_trained = epoch

    return model
