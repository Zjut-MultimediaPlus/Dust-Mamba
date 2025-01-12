import argparse
import datetime
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import wandb
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn import functional as func
from torch import optim
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.detect_dataloader_strategies_3 import LSDSSIMRDataset

from utils.Dust_Mamba import Dust_Mamba, Add_MRDF, Add_VSB, Dust_Mamba_joint_training
from utils.baselines import NestedUNet, UNet, AttUNet
from utils.evaluation_average import multiclass_dice_coeff, mKappa, mPOD_recall, mTAR_precision, mIOU_csi
from utils.evaluation_intensity_levels import multiclass_dice_coeff_intensity_levels, mKappa_intensity_levels, \
    mPOD_recall_intensity_levels, mTAR_precision_intensity_levels, mIOU_csi_intensity_levels
from utils.my_logger import my_logger
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
# from utils.HDF5Dataset import HDF5Dataset
# from utils.mean_std import get_mean_and_std

from utils.vmunet import *
from utils.vmamba import *
from torch.nn.utils import clip_grad_norm_
import torch.backends.cudnn as cudnn
import random

here = os.path.dirname(os.path.abspath(__file__))


def get_base_config(oc_from_file=None):
    oc = OmegaConf.create()
    oc.dataset = get_dataset_config()
    oc.optim = get_optim_config()
    oc.logging = get_logging_config()
    oc.vis = get_vis_config()
    if oc_from_file is not None:
        oc = OmegaConf.merge(oc, oc_from_file)
    return oc


def get_dataset_config():
    oc = OmegaConf.create()
    return oc


def get_optim_config():
    oc = OmegaConf.create()
    return oc


def get_logging_config():
    oc = OmegaConf.create()
    return oc


def get_vis_config():
    oc = OmegaConf.create()
    return oc


# occurrence detection
def test_binary(model, test_loader, device, epoch, iteration):
    logger.info("Starting occurrence detection test...")
    model.eval()
    test_batch_num = len(test_loader)
    dice_score = 0
    iou = 0
    pod = 0
    tar = 0
    kappa = 0

    with tqdm(test_loader, total=test_batch_num, desc=f'Testing epoch:{epoch}/{cfg_oc.optim.epochs}', unit='batch',
              ncols=120) as pbar:
        for data, label_true in test_loader:
            # process label
            label_true[label_true < 12] = 0.
            label_true[label_true >= 12] = 1.
            data = data.to(device=device, dtype=torch.float32)
            label_true = label_true.to(device=device, dtype=torch.long)

            # detection
            with torch.no_grad():
                label_pred, _ = model(data)

                label_pred = func.one_hot(label_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
                label_true = func.one_hot(label_true, 2).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(label_pred[:, 1:, ...], label_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                iou += mIOU_csi(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)
                pod += mPOD_recall(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)
                tar += mTAR_precision(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)
                kappa += mKappa(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)

            pbar.update(1)

    if test_batch_num == 0:
        return dice_score, iou, pod, tar, kappa
    return dice_score / test_batch_num, iou / test_batch_num, pod / test_batch_num, tar / test_batch_num, kappa / test_batch_num


# intensity detection
def test_multi(model, test_loader, device, epoch, iteration):
    logger.info("Starting intensity detection test...")
    model.eval()
    test_batch_num = len(test_loader)
    # val_loss = 0
    dice_score = 0
    iou = 0
    pod = 0
    tar = 0
    kappa = 0
    dice_list_score = [0, 0, 0, 0, 0, 0]
    iou_list = [0, 0, 0, 0, 0, 0]
    pod_list = [0, 0, 0, 0, 0, 0]
    tar_list = [0, 0, 0, 0, 0, 0]
    kappa_list = [0, 0, 0, 0, 0, 0]

    with tqdm(test_loader, total=test_batch_num, desc=f'Testing epoch:{epoch}/{cfg_oc.optim.epochs}', unit='batch',
              ncols=120) as pbar:
        for data, label_true in test_loader:
            # process label
            label_true[label_true < 12] = 0.
            label_true[(label_true >= 12) & (label_true < 15)] = 1.
            label_true[(label_true >= 15) & (label_true < 17)] = 2.
            label_true[(label_true >= 17) & (label_true < 19)] = 3.
            label_true[(label_true >= 19) & (label_true < 21)] = 4.
            label_true[(label_true >= 21) & (label_true < 23)] = 5.
            label_true[(label_true >= 23) & (label_true <= 24)] = 6.
            data = data.to(device=device, dtype=torch.float32)
            label_true = label_true.to(device=device, dtype=torch.long)

            # detection
            with torch.no_grad():
                _, label_pred = model(data)

                label_pred = func.one_hot(label_pred.argmax(dim=1), 7).permute(0, 3, 1, 2).float()
                label_true = func.one_hot(label_true, 7).permute(0, 3, 1, 2).float()

                dice_t, dice_list_t = multiclass_dice_coeff_intensity_levels(label_pred[:, 1:, ...],
                                                                             label_true[:, 1:, ...],
                                                                             reduce_batch_first=False)
                dice_score += dice_t

                iou_t, iou_list_t = mIOU_csi_intensity_levels(label_pred[:, 1:, ...], label_true[:, 1:, ...],
                                                              reduce_batch_first=False)
                iou += iou_t

                pod_t, pod_list_t = mPOD_recall_intensity_levels(label_pred[:, 1:, ...], label_true[:, 1:, ...],
                                                                 reduce_batch_first=False)
                pod += pod_t

                tar_t, tar_list_t = mTAR_precision_intensity_levels(label_pred[:, 1:, ...], label_true[:, 1:, ...],
                                                                    reduce_batch_first=False)
                tar += tar_t

                kappa_t, kappa_list_t = mKappa_intensity_levels(label_pred[:, 1:, ...], label_true[:, 1:, ...],
                                                                reduce_batch_first=False)
                kappa += kappa_t

                for i in range(len(dice_list_score)):
                    dice_list_score[i] += dice_list_t[i]
                    iou_list[i] += iou_list_t[i]
                    pod_list[i] += pod_list_t[i]
                    tar_list[i] += tar_list_t[i]
                    kappa_list[i] += kappa_list_t[i]

            pbar.update(1)

        dice_list_score = [x / test_batch_num for x in dice_list_score]
        iou_list = [x / test_batch_num for x in iou_list]
        pod_list = [x / test_batch_num for x in pod_list]
        tar_list = [x / test_batch_num for x in tar_list]
        kappa_list = [x / test_batch_num for x in kappa_list]


    if test_batch_num == 0:
        return dice_score, iou, pod, tar, kappa
    return dice_score / test_batch_num, iou / test_batch_num, pod / test_batch_num, tar / test_batch_num, kappa / test_batch_num, dice_list_score, iou_list, pod_list, tar_list, kappa_list


def save_colored_image(label, path, title, num_classes=7):
    """
    Save an image with a fixed color mapping.

    :param label: Input label image.
    :param path: Path to save the image.
    :param title: Title of the image.
    :param num_classes: Number of classes (used to select the color map).
    """

    # Define the color mapping
    binary_cmap = ListedColormap(['black', 'orange'])
    multi_cmap = ListedColormap(['black', 'lightblue', 'blue', 'green', 'yellow', 'orange', 'darkred'])

    if num_classes == 2:
        cmap = binary_cmap
        bounds = [-0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap.N)

    elif num_classes == 7:
        cmap = multi_cmap
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        norm = BoundaryNorm(bounds, cmap.N)

    else:
        raise ValueError("num_classes should be either 2 or 7.")

    plt.figure(figsize=(10, 6))
    plt.imshow(label, cmap=cmap, norm=norm, interpolation='nearest')

    plt.title(title)
    plt.axis('off')
    plt.savefig(path)
    plt.close()


def visual_multi(model, test_loader, device, epoch, iteration):
    logger.info("Starting Visualization...")
    model.eval()
    test_batch_num = len(test_loader)

    with tqdm(test_loader, total=test_batch_num, desc=f'Testing epoch:{epoch}/{cfg_oc.optim.epochs}', unit='batch',
              ncols=120) as pbar:
        for data, label_true in test_loader:
            # process label
            label_true[label_true < 12] = 0.
            label_true[(label_true >= 12) & (label_true < 15)] = 1.
            label_true[(label_true >= 15) & (label_true < 17)] = 2.
            label_true[(label_true >= 17) & (label_true < 19)] = 3.
            label_true[(label_true >= 19) & (label_true < 21)] = 4.
            label_true[(label_true >= 21) & (label_true < 23)] = 5.
            label_true[(label_true >= 23) & (label_true <= 24)] = 6.
            data = data.to(device=device, dtype=torch.float32)
            label_true = label_true.to(device=device, dtype=torch.long)

            # detection
            with torch.no_grad():
                _, label_pred = model(data)

                # Convert predictions to numpy
                pred_label = label_pred.argmax(dim=1)[0].cpu().numpy()
                true_label = label_true[0].cpu().numpy()

                # Generate and save continuous color images
                save_colored_image(pred_label, "pred_label.png", "Prediction", num_classes=7)
                save_colored_image(true_label, "true_label.png", "Label", num_classes=7)

                # Log images
                experiment.log({
                    'test_label_pred': wandb.Image("pred_label.png",
                                                   caption=f'test_epoch-{epoch}_iter-{iteration}_idx-0_pred'),
                    'test_label_true': wandb.Image("true_label.png",
                                                   caption=f'test_epoch-{epoch}_iter-{iteration}_idx-0_true'),
                })

            pbar.update(1)

    logger.info("Visualization has been completed.")
    return


def test_net(args, cfg_oc, model, device):
    num_classes = cfg_oc.dataset.num_classes
    batch_size = cfg_oc.optim.batch_size * torch.cuda.device_count()
    alpha = cfg_oc.optim.loss_weight
    img_size = (cfg_oc.dataset.img_height, cfg_oc.dataset.img_width)
    use_sat_channels = cfg_oc.dataset.use_sat_channels
    use_mete_channels = cfg_oc.dataset.use_mete_channels

    lsdssimr_test_dataset = LSDSSIMRDataset(
        dir_out=dir_out,
        use_sat_channels=use_sat_channels,
        use_mete_channels=use_mete_channels,
        img_size=img_size,
        lsdssimr_data_dir=args.dir_hdfdata,
        lsdssimr_catalog=os.path.join(args.dir_csv, 'catalog.csv'),
        start_date=datetime.datetime(*cfg_oc.dataset.train_test_split_date),
        end_date=datetime.datetime(*cfg_oc.dataset.end_date),
        preprocess=True,
        shuffle=False,
        mode='test'
    )
    test_num = len(lsdssimr_test_dataset)
    loader_args = dict(batch_size=batch_size, num_workers=14, pin_memory=True)
    test_loader = DataLoader(lsdssimr_test_dataset, shuffle=False, drop_last=True, **loader_args)

    logger.info(f'''Start testing using model {args.model} on device {device}:
        \tInput channels:  {num_channels}
        \tInput sat channels:  {use_sat_channels}
        \tInput mete channels: {use_mete_channels}
        \tOutput classes:  {num_classes}
        \tBatch size:      {batch_size}
        \tTest size:       {test_num}
        \tDevice:          {device}
        ''')

    epoch = 0
    # occurrence detection
    test_score, test_iou, test_pod, test_tar, test_kappa = test_binary(model, test_loader, device, epoch,
                                                                       iteration=0)
    logger.info(f'Test Epoch {epoch}:'
                f'Test Dice_score: {test_score}, '
                f'Test mIOU_CSI: {test_iou}, '
                f'Test mPOD_recall: {test_pod}, '
                f'Test mTAR_precision: {test_tar}, '
                f'Test mKappa: {test_kappa}')
    # intensity detection
    test_score, test_iou, test_pod, test_tar, test_kappa, test_dice_list, test_iou_list, test_pod_list, test_tar_list, test_kappa_list \
        = test_multi(model, test_loader, device, epoch, iteration=0)
    logger.info(f'Test Epoch {epoch}:'
                f'Test Dice_score: {test_score}, '
                f'Test mIOU_CSI: {test_iou}, '
                f'Test mPOD_recall: {test_pod}, '
                f'Test mTAR_precision: {test_tar}, '
                f'Test mKappa: {test_kappa},\n'
                f'Test dice list: {test_dice_list},\n'
                f'Test iou list: {test_iou_list},\n'
                f'Test pod list: {test_pod_list},\n'
                f'Test tar list: {test_tar_list},\n'
                f'Test kappa list: {test_kappa_list},')
    # Visualization
    if cfg_oc.optim.batch_size==1:
        visual_multi(model, test_loader, device, epoch=0)


def get_args():
    parser = argparse.ArgumentParser(description='Train the Transformer UNets on multichannel data and masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='AttUNet', help='Model used for training')
    parser.add_argument("--save", default="tmp_lsdssimr", type=str)
    parser.add_argument('--cfg', type=str, metavar='FILE', default=None, help='path to config file')
    parser.add_argument("--test", action="store_true")
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether use the pretrain parameters')
    parser.add_argument('--load', type=str, default=False, help='Load model form a .pth file')
    parser.add_argument('--online', action='store_true', default=False, help='Path to the directory of HDF files')
    parser.add_argument('--dir_hdfdata', type=str, default=False, help='Path to the directory of HDF files')
    parser.add_argument('--dir_csv', type=str, default=False, help='Path to the directory of csv file')
    parser.add_argument('--dir_checkpoint', type=str, default=False, help='Path to the directory of checkpoint')

    return parser.parse_args()


if __name__ == "__main__":
    # args
    args = get_args()
    if args.test and args.load is None:
        raise ValueError("test mode but checkpoint.pth is None!")

    # dir_out
    dir_out = os.path.join(here, 'experiments', args.model, args.save,
                           datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(dir_out, exist_ok=True)

    # checkpoints
    dir_ckpt = os.path.join(dir_out, 'checkpoints')
    os.makedirs(dir_ckpt)

    # args
    if args.cfg is not None:
        cfg_oc = OmegaConf.load(open(args.cfg, 'r'))
    else:
        cfg_oc = get_base_config(oc_from_file=None)

    # CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg_oc.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = torch.cuda.is_available()

    # dir_hdfdata = '/opt/data/private/'
    # channels
    num_channels = len(cfg_oc.dataset.use_sat_channels) + len(cfg_oc.dataset.use_mete_channels)

    # initialize logging and wandb
    logger = my_logger(os.path.join(dir_out, 'logging.log'), level=logging.INFO)
    experiment = wandb.init(config=args.__dict__,
                            project='New Runs-Various UNets of Dust Detection',
                            resume='allow',
                            notes=f'run {args.model}',
                            anonymous='must',
                            mode='online' if args.online else 'offline',
                            dir=dir_out,
                            name=f'{args.model}',
                            save_code=True)
    experiment.name = f'{args.model}-run--{experiment.id}'
    logger.info(args.__dict__)
    logger.info(cfg_oc.__dict__.get("_content"))

    # model: joint training
    model = Dust_Mamba_joint_training(num_channels, cfg_oc.dataset.num_classes, cfg_oc.optim.bilinear)
    filename = args.dir_checkpoint
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint)


    if cuda:
        model.to(device=device)

    # test
    test_net(args=args, cfg_oc=cfg_oc, model=model, device=device)
