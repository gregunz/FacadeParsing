import argparse
import datetime
import json
import os

import pytz
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from facade_project import PATH_TO_DATA, LABEL_NAME_TO_VALUE, \
    FACADE_ROT_IMAGES_TENSORS_DIR, FACADE_ROT_HEATMAPS_TENSORS_DIR, FACADE_ROT_MEAN, FACADE_ROT_STD, CROP_SIZE, \
    DEFAULT_SEED_SPLIT
from facade_project.data import FacadeRandomRotDataset, TransformedDataset, split, to_dataloader
from facade_project.data.augmentation import random_brightness_and_contrast, random_crop, random_flip, compose
from facade_project.nn.losses import facade_criterion
from facade_project.nn.metrics import FacadeMetric
from facade_project.nn.models import UNet, AlbuNet
from facade_project.nn.train import train_model


def run_name(model_name, predictions, pred_weights):
    timezone = pytz.timezone(pytz.country_timezones['ch'][0])
    date_time_str = datetime.datetime.now(tz=timezone).strftime("%Y-%m-%d_%H-%M-%S")

    pred_str_list = ['{}{}'.format(p, w) for p, w in zip(predictions, pred_weights)]

    return '{}_{}_predictions-{}'.format(
        date_time_str,
        model_name,
        '-'.join(pred_str_list),
    )


def main(args):
    assert len(args.predictions) > 0
    assert args.epochs > 0
    assert args.batch_train > 0
    assert args.batch_val > 0

    label_name_to_value = LABEL_NAME_TO_VALUE
    # if args.labels is not None:
    #    label_name_to_value = {n:v for n,v in label_name_to_value.items() if v in args.labels}

    if args.pred_weights is None:
        args.pred_weights = [1.0 for _ in args.predictions]
    #assert sum(args.pred_weights) == 1

    device = torch.device(args.device)

    run_name_str = run_name(
        model_name=args.model,
        predictions=args.predictions,
        pred_weights=args.pred_weights,
    )
    print('Run named {} started...'.format(run_name_str))
    # make directory for model weights
    weights_dir_path = '{}/{}'.format(args.path_for_weights, run_name_str)
    os.mkdir(weights_dir_path)
    summary_dict = vars(args)
    json.dump(
        obj=summary_dict,
        fp=open('{}/summary.json'.format(weights_dir_path), mode='w'),
        sort_keys=True,
        indent=4
    )

    with_heatmaps = 'heatmaps' in args.predictions

    def create_heatmaps(img_idx, rot_idx, device):
        if not with_heatmaps:
            return dict()

        def get_filename(idx, jdx):
            return '{}/heatmaps_door-window_{:03d}_{:03d}.torch' \
                .format(FACADE_ROT_HEATMAPS_TENSORS_DIR, idx, jdx)

        return {
            'heatmaps': torch.load(get_filename(img_idx, rot_idx), map_location=device)
        }

    facade_dataset = FacadeRandomRotDataset(
        img_dir=FACADE_ROT_IMAGES_TENSORS_DIR,
        add_aux_channels_fn=create_heatmaps,
        img_to_num_rot=None,
        caching=False,  # when true it takes quite a lot of RAM
        init_caching=False,
        device=device,
    )

    mean = torch.tensor(FACADE_ROT_MEAN).view(3, 1, 1).to(device)
    std = torch.tensor(FACADE_ROT_STD).view(3, 1, 1).to(device)
    tf = compose([
        random_crop(crop_size=args.crop_size),  # Cropping
        lambda img, lbl: (random_brightness_and_contrast()(img), lbl),  # Augmentation
        random_flip(),  # Flipping (augmentation)
        lambda img, lbl: ((img - mean) / std, lbl)
    ])

    transformed_facade_dataset = TransformedDataset(
        dataset=facade_dataset,
        transform=tf,
    )

    dataset_train, dataset_val = split(transformed_facade_dataset, seed=args.split_seed)

    dataloaders = {'train': to_dataloader(dataset_train, args.batch_train),
                   'val': to_dataloader(dataset_val, args.batch_val)}

    num_target_channels = 0
    for p in args.predictions:
        if p == 'mask':
            num_target_channels += len(label_name_to_value)
        elif p == 'heatmaps':
            num_target_channels += 3  # heatmaps for center, width and height
    assert num_target_channels > 0, 'model should predict at least one thing'

    if args.model == 'albunet':
        model = AlbuNet(
            num_classes=num_target_channels,
            num_filters=16,
            pretrained=False,
            is_deconv=True,
        )
    elif args.model == 'unet':
        model = UNet(
            in_channels=3,
            n_classes=num_target_channels,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv'
        )
    else:
        raise Exception('model undefined')

    optimizer = optim.Adam(model.parameters())
    criterion = facade_criterion(
        predictions_list=args.predictions,
        predictions_weights=args.pred_weights,
        num_classes=len(label_name_to_value),
        use_dice=args.use_dice,
        center_factor=args.center_factor,
    )
    with torch.cuda.device(device):
        model = train_model(
            dataloaders=dataloaders,
            path_weights=args.path_for_weights,
            model_name=run_name_str,
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1),
            metric_handler=FacadeMetric(args.predictions, label_name_to_value),
            writer=SummaryWriter('runs/{}'.format(run_name_str)),
            num_epoch=args.epochs,
            keep_n_best=10,
            verbose=True,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to perform facade parsing')

    parser.add_argument('--model', action="store", dest="model", type=str, default='albunet')
    parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=100)
    parser.add_argument('--split-seed', action="store", dest="split_seed", type=int, default=DEFAULT_SEED_SPLIT)
    parser.add_argument('--batch-train', action="store", dest="batch_train", type=int, default=1)
    parser.add_argument('--batch-val', action="store", dest="batch_val", type=int, default=1)
    parser.add_argument('--predictions', action='store', dest='predictions', nargs='+', type=str,
                        default=['mask', 'heatmaps'])
    parser.add_argument('--pred-weights', action='store', dest='pred_weights', nargs='+', type=float, default=None)
    parser.add_argument('--path-for-weights', action='store', dest='path_for_weights', type=str,
                        default='{}/models'.format(PATH_TO_DATA))
    parser.add_argument('--device', action='store', dest='device', type=str, default='cuda:0')
    parser.add_argument('--use-dice', action='store', dest='use_dice', type=bool, default=True)
    parser.add_argument('--crop-size', action='store', dest='crop_size', type=int, default=CROP_SIZE)
    parser.add_argument('--center-factor', action='store', dest='center_factor', type=float, default=100.0)

    main(parser.parse_args())
