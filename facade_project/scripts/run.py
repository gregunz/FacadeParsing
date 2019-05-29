import argparse
import datetime
import json

import pytz
import torch.optim as optim
from tensorboardX import SummaryWriter

from facade_project import SIGMA_FIXED, IS_SIGMA_FIXED, SIGMA_SCALE, PATH_TO_DATA, LABEL_NAME_TO_VALUE, \
    FACADE_ROT_HEATMAPS_INFOS_PATH, FACADE_ROT_DIR
from facade_project.data import FacadeDatasetRandomRot, TransformedDataset, split, to_dataloader
from facade_project.data.augmentation import random_brightness_and_contrast, random_crop, random_flip, compose
from facade_project.geometry.heatmap import build_heatmaps
from facade_project.nn.losses import facade_criterion
from facade_project.nn.models import UNet, AlbuNet
from facade_project.nn.train import train_model


def run_name(model_name, predictions, is_sigma_fixed, sigma_fixed, sigma_scale):
    timezone = pytz.timezone(pytz.country_timezones['ch'][0])
    date_time_str = datetime.datetime.now(tz=timezone).strftime("%Y-%m-%d_%H-%M-%S")

    sigma_str = ''
    if 'center' in predictions or 'size' in predictions:
        sigma_str = '_fixed-{}'.format(sigma_fixed) if is_sigma_fixed else 'scale{}'.format(sigma_scale),

    return '{}_{}_predictions{}{}'.format(
        model_name,
        date_time_str,
        '-'.join(predictions),
        sigma_str
    )


def main(args):
    heatmaps_infos_per_rot = json.load(open(FACADE_ROT_HEATMAPS_INFOS_PATH, mode='r'))

    def create_heatmaps(img_idx, rot_idx):
        info = heatmaps_infos_per_rot[img_idx][rot_idx]
        return build_heatmaps(
            heatmap_info=info,
            label_name_to_value=LABEL_NAME_TO_VALUE,
            is_sigma_fixed=args.is_sigma_fixed,
            sigma_fixed=args.sigma_fixed,
            sigma_scale=args.sigma_scale,
            heatmap_types=args.predictions
        )

    with_heatmaps = 'center' in args.predictions or 'size' in args.predictions

    facade_dataset = FacadeDatasetRandomRot(
        img_dir=FACADE_ROT_DIR,
        add_aux_channels_fn=create_heatmaps if with_heatmaps else None,
        img_to_num_rot=None,
        caching=True,  # one should check whether this uses too much RAM
        init_caching=False,
    )

    tf = compose([
        lambda img, lbl: random_crop(img, lbl, crop_size=768),  # Cropping
        lambda img, lbl: (random_brightness_and_contrast(img), lbl),  # Augmentation
        random_flip,  # Flipping (augmentation)
    ])

    transformed_facade_dataset = TransformedDataset(
        dataset=facade_dataset,
        transform=tf,
    )

    dataset_train, dataset_val = split(transformed_facade_dataset, seed=args.split_seed)

    dataloaders = {'train': to_dataloader(dataset_train, args.batch_train),
                   'val': to_dataloader(dataset_val, args.batch_val)}

    num_classes = len(args.heatmaps) + (1 if args.include_mask else 0)
    assert num_classes > 0, 'model should predict at least one thing'

    if args.model == 'albunet':
        model = AlbuNet(
            num_classes=num_classes,
            num_filters=16,
            pretrained=True,
            is_deconv=True,
        )
    elif args.model == 'unet':
        model = UNet(
            in_channels=3,
            n_classes=num_classes,
            padding=True,
            batch_norm=True,
            up_mode='upconv'
        )
    else:
        raise Exception('model undefined')

    run_name_str = run_name(
        model_name=args.model,
        predictions=args.predictions,
        is_sigma_fixed=args.is_sigma_fixed,
        sigma_fixed=args.sigma_fixed,
        sigma_scale=args.sigma_scaled,
    )

    optimizer = optim.Adam(model.parameters())
    model = train_model(
        dataloaders=dataloaders,
        path_to_data=args.path_for_weights,
        model_name=run_name_str,
        model=model,
        device=args.device,
        criterion=facade_criterion,
        optimizer=optimizer,
        scheduler=None,  # optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1),
        writer=SummaryWriter('runs/{}.log'.format(run_name_str)),
        num_epoch=args.epochs,
        keep_n_best=3,
        verbose=True,
        label_name_to_value=LABEL_NAME_TO_VALUE
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to perform facade parsing')

    parser.add_argument('--model', action="store", dest="model", type=str, default='albunet')
    # parser.add_argument('--num-filters', action="store", dest="num_filters", type=int, default=16)
    # parser.add_argument('--pretrained', action="store", dest="pretrained", type=bool, default=True)
    parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=100)
    parser.add_argument('--split-seed', action="store", dest="split_seed", type=int, default=238122)
    parser.add_argument('--batch-train', action="store", dest="batch_train", type=int, default=1)
    parser.add_argument('--batch-val', action="store", dest="batch_test", type=int, default=1)
    parser.add_argument('--predictions', action='store', dest='predictions', nargs='+', type=str,
                        default=['center', 'size', 'mask'])
    parser.add_argument('--is-sigma-fixed', action='store', dest='is_sigma_fixed', type=bool, default=IS_SIGMA_FIXED)
    parser.add_argument('--sigma-fixed', action='store', dest='sigma_fixed', type=float, default=SIGMA_FIXED)
    parser.add_argument('--sigma-scale', action='store', dest='sigma_scale', type=float, default=SIGMA_SCALE)
    parser.add_argument('--path-for-weights', action='store', dest='path_for_weights', type=str, default=PATH_TO_DATA)
    parser.add_argument('--device', action='store', dest='device', type=str, default='cuda:0')
    # parser.add_argument('--', action='store', dest='', type=, default=)

    main(parser.parse_args())
