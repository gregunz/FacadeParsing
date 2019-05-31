import argparse
import datetime

import pytz
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from facade_project import SIGMA_FIXED, IS_SIGMA_FIXED, SIGMA_SCALE, PATH_TO_DATA, LABEL_NAME_TO_VALUE, \
    FACADE_ROT_DIR
from facade_project.data import FacadeRandomRotDataset, TransformedDataset, split, to_dataloader
from facade_project.data.augmentation import random_brightness_and_contrast, random_crop, random_flip, compose
from facade_project.geometry.heatmap import build_heatmaps
from facade_project.nn.losses import facade_criterion
from facade_project.nn.metrics import FacadeMetric
from facade_project.nn.models import UNet, AlbuNet
from facade_project.nn.train import train_model
from facade_project.utils.load import HEATMAP_INFOS_PER_ROT


def run_name(model_name, predictions, is_sigma_fixed, sigma_fixed, sigma_scale):
    timezone = pytz.timezone(pytz.country_timezones['ch'][0])
    date_time_str = datetime.datetime.now(tz=timezone).strftime("%Y-%m-%d_%H-%M-%S")

    sigma_str = ''
    if 'center' in predictions or 'size' in predictions:
        if is_sigma_fixed:
            sigma_str = '_fixed-{}'.format(sigma_fixed)
        else:
            sigma_str = '_scale-{}'.format(sigma_scale)

    return '{}_{}_predictions{}{}'.format(
        model_name,
        date_time_str,
        '-'.join(predictions),
        sigma_str
    )


def main(args):
    # args.predictions = list(set(args.predictions))
    assert len(args.predictions) == len(args.pred_weights)
    assert args.epochs > 0
    assert args.batch_train > 0
    assert args.batch_val > 0

    device = torch.device(args.device)

    def create_heatmaps(img_idx, rot_idx):
        info = HEATMAP_INFOS_PER_ROT[img_idx][rot_idx]
        return build_heatmaps(
            heatmap_info=info,
            label_name_to_value=LABEL_NAME_TO_VALUE,
            is_sigma_fixed=args.is_sigma_fixed,
            sigma_fixed=args.sigma_fixed,
            sigma_scale=args.sigma_scale,
            heatmap_types=args.predictions
        )

    with_heatmaps = 'center' in args.predictions \
                    or 'width' in args.predictions \
                    or 'height' in args.predictions

    facade_dataset = FacadeRandomRotDataset(
        img_dir=FACADE_ROT_DIR,
        add_aux_channels_fn=create_heatmaps if with_heatmaps else None,
        img_to_num_rot=None,
        caching=True,  # one should check whether this uses too much RAM
        init_caching=False,
    )

    tf = compose([
        random_crop(crop_size=768),  # Cropping
        lambda img, lbl: (random_brightness_and_contrast()(img), lbl),  # Augmentation
        random_flip(),  # Flipping (augmentation)
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
            num_target_channels += len(LABEL_NAME_TO_VALUE)
        elif p == 'center' or p == 'width' or p == 'height':
            num_target_channels += len(LABEL_NAME_TO_VALUE) - 1  # no heatmap for background
    assert num_target_channels > 0, 'model should predict at least one thing'

    if args.model == 'albunet':
        model = AlbuNet(
            num_classes=num_target_channels,
            num_filters=16,
            pretrained=True,
            is_deconv=True,
        )
    elif args.model == 'unet':
        model = UNet(
            in_channels=3,
            n_classes=num_target_channels,
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
        sigma_scale=args.sigma_scale,
    )

    optimizer = optim.Adam(model.parameters())
    criterion = facade_criterion(
        predictions_list=args.predictions,
        predictions_weights=args.pred_weights,
        num_classes=len(LABEL_NAME_TO_VALUE),
        use_dice=True
    )
    model = train_model(
        dataloaders=dataloaders,
        path_weights=args.path_for_weights,
        model_name=run_name_str,
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,  # optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1),
        metric_handler=FacadeMetric(args.predictions, LABEL_NAME_TO_VALUE),
        writer=SummaryWriter('runs/{}.log'.format(run_name_str)),
        num_epoch=args.epochs,
        keep_n_best=3,
        verbose=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to perform facade parsing')

    parser.add_argument('--model', action="store", dest="model", type=str, default='albunet')
    # parser.add_argument('--num-filters', action="store", dest="num_filters", type=int, default=16)
    # parser.add_argument('--pretrained', action="store", dest="pretrained", type=bool, default=True)
    parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=100)
    parser.add_argument('--split-seed', action="store", dest="split_seed", type=int, default=238122)
    parser.add_argument('--batch-train', action="store", dest="batch_train", type=int, default=1)
    parser.add_argument('--batch-val', action="store", dest="batch_val", type=int, default=1)
    parser.add_argument('--predictions', action='store', dest='predictions', nargs='+', type=str,
                        default=['mask', 'center', 'height', 'width'])
    parser.add_argument('--pred-weights', action='store', dest='pred_weights', nargs='+', type=float,
                        default=[1., 1., 1., 1.])
    parser.add_argument('--is-sigma-fixed', action='store', dest='is_sigma_fixed', type=bool, default=IS_SIGMA_FIXED)
    parser.add_argument('--sigma-fixed', action='store', dest='sigma_fixed', type=float, default=SIGMA_FIXED)
    parser.add_argument('--sigma-scale', action='store', dest='sigma_scale', type=float, default=SIGMA_SCALE)
    parser.add_argument('--path-for-weights', action='store', dest='path_for_weights', type=str,
                        default='{}/models'.format(PATH_TO_DATA))
    parser.add_argument('--device', action='store', dest='device', type=str, default='cuda:0')
    # parser.add_argument('--', action='store', dest='', type=, default=)

    main(parser.parse_args())
