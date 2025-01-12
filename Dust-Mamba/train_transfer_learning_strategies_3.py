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
from utils.evaluation_average import dice_loss, multiclass_dice_coeff, mKappa, mPOD_recall, mTAR_precision, mIOU_csi
from utils.my_logger import my_logger

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
def validation_binary(epoch, model, val_loader, device):
    model.eval()
    val_batch_num = len(val_loader)
    # val_loss = 0
    dice_score = 0
    iou = 0
    pod = 0
    tar = 0
    kappa = 0

    with tqdm(val_loader, total=val_batch_num, desc=f'Validation round of epoch {epoch}', unit='batch', leave=True,
              ncols=100) as pbar:
        for data, label_true in val_loader:
            # process label
            label_true[label_true < 12] = 0.
            label_true[label_true >= 12] = 1.
            data = data.to(device=device, dtype=torch.float32)
            label_true = label_true.to(device=device, dtype=torch.long)
            label_true = func.one_hot(label_true, 2).permute(0, 3, 1, 2).float()

            # detection
            with torch.no_grad():
                label_pred, _ = model(data)
                label_pred = func.one_hot(label_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(label_pred[:, 1:, ...], label_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                iou += mIOU_csi(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)
                pod += mPOD_recall(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)
                tar += mTAR_precision(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)
                kappa += mKappa(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)

            pbar.update(1)

    model.train()
    if val_batch_num == 0:
        return dice_score, iou, pod, tar, kappa
    return dice_score / val_batch_num, iou / val_batch_num, pod / val_batch_num, tar / val_batch_num, kappa / val_batch_num


# intensity detection
def validation_multi(epoch, model, val_loader, device):
    model.eval()
    val_batch_num = len(val_loader)
    # val_loss = 0
    dice_score = 0
    iou = 0
    pod = 0
    tar = 0
    kappa = 0

    with tqdm(val_loader, total=val_batch_num, desc=f'Validation round of epoch {epoch}', unit='batch', leave=True,
              ncols=100) as pbar:
        for data, label_true in val_loader:
            data = data.to(device=device, dtype=torch.float32)
            # process label
            label_true[label_true < 12] = 0.
            label_true[(label_true >= 12) & (label_true < 15)] = 1.
            label_true[(label_true >= 15) & (label_true < 17)] = 2.
            label_true[(label_true >= 17) & (label_true < 19)] = 3.
            label_true[(label_true >= 19) & (label_true < 21)] = 4.
            label_true[(label_true >= 21) & (label_true < 23)] = 5.
            label_true[(label_true >= 23) & (label_true <= 24)] = 6.
            label_true = label_true.to(device=device, dtype=torch.long)
            label_true = func.one_hot(label_true, 7).permute(0, 3, 1, 2).float()

            # detection
            with torch.no_grad():
                _, label_pred = model(data)
                label_pred = func.one_hot(label_pred.argmax(dim=1), 7).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(label_pred[:, 1:, ...], label_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                iou += mIOU_csi(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)
                pod += mPOD_recall(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)
                tar += mTAR_precision(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)
                kappa += mKappa(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)

            pbar.update(1)
    model.train()

    # dice score
    if val_batch_num == 0:
        return dice_score, iou, pod, tar, kappa
    return dice_score / val_batch_num, iou / val_batch_num, pod / val_batch_num, tar / val_batch_num, kappa / val_batch_num


# occurrence detection
def test_binary(model, test_loader, device, epoch, iteration):
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

    model.train()

    if test_batch_num == 0:
        return dice_score, iou, pod, tar, kappa
    return dice_score / test_batch_num, iou / test_batch_num, pod / test_batch_num, tar / test_batch_num, kappa / test_batch_num


# intensity detection
def test_multi(model, test_loader, device, epoch, iteration):
    model.eval()
    test_batch_num = len(test_loader)
    # val_loss = 0
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
                dice_score += multiclass_dice_coeff(label_pred[:, 1:, ...], label_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                iou += mIOU_csi(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)
                pod += mPOD_recall(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)
                tar += mTAR_precision(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)
                kappa += mKappa(label_pred[:, 1:, ...], label_true[:, 1:, ...], reduce_batch_first=False)

            pbar.update(1)

    model.train()

    if test_batch_num == 0:
        return dice_score, iou, pod, tar, kappa
    return dice_score / test_batch_num, iou / test_batch_num, pod / test_batch_num, tar / test_batch_num, kappa / test_batch_num


def train_net(args, cfg_oc, model, device):
    lr = cfg_oc.optim.lr
    epochs = cfg_oc.optim.epochs
    num_classes = cfg_oc.dataset.num_classes
    batch_size = cfg_oc.optim.batch_size * torch.cuda.device_count()
    alpha = cfg_oc.optim.loss_weight
    img_size = (cfg_oc.dataset.img_height, cfg_oc.dataset.img_width)
    use_sat_channels = cfg_oc.dataset.use_sat_channels
    use_mete_channels = cfg_oc.dataset.use_mete_channels
    iteration = 0

    # occurrence detection
    test_score_sum_1 = 0
    test_iou_sum_1 = 0
    test_pod_sum_1 = 0
    test_tar_sum_1 = 0
    test_kappa_sum_1 = 0

    # intensity detection
    test_score_sum_2 = 0
    test_iou_sum_2 = 0
    test_pod_sum_2 = 0
    test_tar_sum_2 = 0
    test_kappa_sum_2 = 0
    # max_iterations = args.max_iterations

    # 1. dataset
    lsdssimr_train_dataset = LSDSSIMRDataset(
        dir_out=dir_out,
        use_sat_channels=use_sat_channels,
        use_mete_channels=use_mete_channels,
        img_size=img_size,
        lsdssimr_data_dir=args.dir_hdfdata,
        lsdssimr_catalog=os.path.join(args.dir_csv, 'catalog.csv'),
        start_date=datetime.datetime(*cfg_oc.dataset.start_date),
        end_date=datetime.datetime(*cfg_oc.dataset.train_val_split_date),
        preprocess=True,
        shuffle=False,
        mode='train'
    )
    lsdssimr_val_dataset = LSDSSIMRDataset(
        dir_out=dir_out,
        use_sat_channels=use_sat_channels,
        use_mete_channels=use_mete_channels,
        img_size=img_size,
        lsdssimr_data_dir=args.dir_hdfdata,
        lsdssimr_catalog=os.path.join(args.dir_csv, 'catalog.csv'),
        start_date=datetime.datetime(*cfg_oc.dataset.train_val_split_date),
        end_date=datetime.datetime(*cfg_oc.dataset.train_test_split_date),
        preprocess=True,
        shuffle=False,
        mode='val'
    )
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

    train_num = len(lsdssimr_train_dataset)
    val_num = len(lsdssimr_val_dataset)
    test_num = len(lsdssimr_test_dataset)
    loader_args = dict(batch_size=batch_size, num_workers=14, pin_memory=True)
    train_loader = DataLoader(lsdssimr_train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(lsdssimr_val_dataset, shuffle=False, drop_last=False, **loader_args)
    test_loader = DataLoader(lsdssimr_test_dataset, shuffle=False, drop_last=True, **loader_args)

    logger.info(f'''Start training using model {args.model} on device {device}:
    \tEpochs:          {epochs}
    \tInput channels:  {num_channels}
    \tInput sat channels:  {use_sat_channels}
    \tInput mete channels: {use_mete_channels}
    \tOutput classes:  {num_classes}
    \tBatch size:      {batch_size}
    \tLearning rate:   {lr}
    \tTraining size:   {train_num}
    \tValidation size: {val_num}
    \tTest size:       {test_num}
    \tSave checkpoint: {cfg_oc.save_checkpoint}
    \tDevice:          {device}
    \tMixed precision: {cfg_oc.optim.amp}
    ''')

    # 2. optimization
    max_iteration = epochs * len(train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg_oc.optim.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iteration)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg_oc.optim.amp)
    criterion_ce = nn.CrossEntropyLoss()

    # 3. training
    val_max_1 = 0
    val_max_2 = 0
    experiment.watch(model)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=train_num, desc=f'Epoch {epoch}/{epochs}', unit='images', ncols=120) as pbar:
            for batch_idx, (images, labels_true) in enumerate(train_loader):
                iteration += 1
                assert images.shape[1] == model.in_channels, \
                    f'Network {args.model} has {num_channels} input channels,' \
                    f'but loaded multichannel data have {images.shape[1]} channels. ' \
                    f'Please check that the data are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                labels_true = labels_true.to(device=device, dtype=torch.long)

                labels_binary = labels_true.clone()
                labels_multi = labels_true.clone()

                # Process binary classification labels
                labels_binary[labels_binary < 12] = 0.
                labels_binary[labels_binary >= 12] = 1.

                # Process multi-class labels
                labels_multi[labels_multi < 12] = 0.
                labels_multi[(labels_multi >= 12) & (labels_multi < 15)] = 1.
                labels_multi[(labels_multi >= 15) & (labels_multi < 17)] = 2.
                labels_multi[(labels_multi >= 17) & (labels_multi < 19)] = 3.
                labels_multi[(labels_multi >= 19) & (labels_multi < 21)] = 4.
                labels_multi[(labels_multi >= 21) & (labels_multi < 23)] = 5.
                labels_multi[(labels_multi >= 23) & (labels_multi <= 24)] = 6.

                # Convert labels to integer type
                labels_binary = labels_binary.to(dtype=torch.long)
                labels_multi = labels_multi.to(dtype=torch.long)

                # loss
                with torch.cuda.amp.autocast(enabled=cfg_oc.optim.amp):
                    labels_pred_1, labels_pred_2 = model(images)

                    loss_1 = 10 * ((1 - alpha) * criterion_ce(labels_pred_1, labels_binary) + alpha * dice_loss(
                        func.softmax(labels_pred_1, dim=1).float(),
                        func.one_hot(labels_binary, 2).permute(0, 3, 1, 2).float(), multiclass=True))

                    loss_2 = 10 * ((1 - alpha) * criterion_ce(labels_pred_2, labels_multi) + alpha * dice_loss(
                        func.softmax(labels_pred_2, dim=1).float(),
                        func.one_hot(labels_multi, 7).permute(0, 3, 1, 2).float(), multiclass=True))

                    loss = 0.5 * loss_1 + 0.5 * loss_2

                    if torch.isnan(loss):
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f'current learning rate={current_lr}')
                        print('loss is nan while training!!!!')
                        raise ValueError('loss is nan while training')

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()

                grad_scaler.step(optimizer)
                grad_scaler.update()
                epoch_loss += loss.item()

                pbar.update(batch_size)

                experiment.log({
                    'train loss': loss.item(),
                    'iteration': iteration,
                    'epoch': epoch
                })

                pbar.set_postfix(**{'batch loss': loss.item()})

                scheduler.step()

        histograms = {}
        # 4. validate
        # occurrence detection
        val_score, iou, pod, tar, kappa = validation_binary(epoch, model, val_loader, device)
        experiment.log({
            'epoch': epoch,
            'iteration': iteration,
            'learning rate': optimizer.param_groups[0]['lr'],
            'mDice score': val_score,
            'mIOU_CSI': iou,
            'mPOD_recall': pod,
            'mTAR_precision': tar,
            'mKappa': kappa,
            **histograms
        })
        logger.info(f'Epoch {epoch}/{epochs}:'
                    f'mDice_score: {val_score}, '
                    f'mIOU_CSI: {iou}, '
                    f'mPOD_recall: {pod}, '
                    f'mTAR_precision: {tar}, '
                    f'mKappa: {kappa}')
        if val_score > val_max_1:
            val_max_1 = val_score
            torch.save(model.state_dict(), os.path.join(dir_ckpt, f'checkpoint-{epoch}-binary-best.pth'))
            logger.info(f'Checkpoint of best binary val score on epoch {epoch} has saved.')
        if cfg_oc.save_checkpoint:
            os.makedirs(dir_ckpt, exist_ok=True)
            if epoch > epochs - 5:
                torch.save(model.state_dict(), os.path.join(dir_ckpt, f'checkpoint-epoch{epoch}.pth'))
                logger.info(f'Checkpoint of epoch: {epoch} has saved.')

        # intensity detection
        val_score, iou, pod, tar, kappa = validation_multi(epoch, model, val_loader, device)
        experiment.log({
            'epoch': epoch,
            'iteration': iteration,
            'learning rate': optimizer.param_groups[0]['lr'],
            'mDice score': val_score,
            'mIOU_CSI': iou,
            'mPOD_recall': pod,
            'mTAR_precision': tar,
            'mKappa': kappa,
            **histograms
        })
        logger.info(f'Epoch {epoch}/{epochs}:'
                    f'mDice_score: {val_score}, '
                    f'mIOU_CSI: {iou}, '
                    f'mPOD_recall: {pod}, '
                    f'mTAR_precision: {tar}, '
                    f'mKappa: {kappa}')
        if val_score > val_max_2:
            val_max_2 = val_score
            torch.save(model.state_dict(), os.path.join(dir_ckpt, f'checkpoint-{epoch}-multi-best.pth'))
            logger.info(f'Checkpoint of best multi val score on epoch {epoch} has saved.')
        if cfg_oc.save_checkpoint:
            os.makedirs(dir_ckpt, exist_ok=True)
            if epoch > epochs - 5:
                torch.save(model.state_dict(), os.path.join(dir_ckpt, f'checkpoint-epoch{epoch}.pth'))
                logger.info(f'Checkpoint of epoch: {epoch} has saved.')

        # 5. test
        if epoch > epochs - 5:
            # occurrence detection
            test_score, test_iou, test_pod, test_tar, test_kappa = test_binary(model, test_loader, device, epoch,
                                                                               iteration)
            logger.info(f'Test Epoch {epoch}/{epochs}:'
                        f'Test Dice_score: {test_score}, '
                        f'Test mIOU_CSI: {test_iou}, '
                        f'Test mPOD_recall: {test_pod}, '
                        f'Test mTAR_precision: {test_tar}, '
                        f'Test mKappa: {test_kappa}')
            test_score_sum_1 += test_score
            test_iou_sum_1 += test_iou
            test_pod_sum_1 += test_pod
            test_tar_sum_1 += test_tar
            test_kappa_sum_1 += test_kappa
            # intensity detection
            test_score, test_iou, test_pod, test_tar, test_kappa = test_multi(model, test_loader, device, epoch,
                                                                              iteration)
            logger.info(f'Test Epoch {epoch}/{epochs}:'
                        f'Test Dice_score: {test_score}, '
                        f'Test mIOU_CSI: {test_iou}, '
                        f'Test mPOD_recall: {test_pod}, '
                        f'Test mTAR_precision: {test_tar}, '
                        f'Test mKappa: {test_kappa}')
            test_score_sum_2 += test_score
            test_iou_sum_2 += test_iou
            test_pod_sum_2 += test_pod
            test_tar_sum_2 += test_tar
            test_kappa_sum_2 += test_kappa

    # 6. finish
    logger.info(f'\nTesting binary result:\n'
                f'Test Dice score: {test_score_sum_1 / 5}, '
                f'Test IOU_CSI: {test_iou_sum_1 / 5}, '
                f'Test POD_recall: {test_pod_sum_1 / 5}, '
                f'Test TAR_precision: {test_tar_sum_1 / 5}, '
                f'Test Kappa: {test_kappa_sum_1 / 5}')

    logger.info(f'\nTesting multi result:\n'
                f'Test Dice score: {test_score_sum_2 / 5}, '
                f'Test IOU_CSI: {test_iou_sum_2 / 5}, '
                f'Test POD_recall: {test_pod_sum_2 / 5}, '
                f'Test TAR_precision: {test_tar_sum_2 / 5}, '
                f'Test Kappa: {test_kappa_sum_2 / 5}')

    logger.info('Training Finished!')


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
        \tUsing ckpt:          {args.load}
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
    test_score, test_iou, test_pod, test_tar, test_kappa = test_multi(model, test_loader, device, epoch,
                                                                      iteration=0)
    logger.info(f'Test Epoch {epoch}:'
                f'Test Dice_score: {test_score}, '
                f'Test mIOU_CSI: {test_iou}, '
                f'Test mPOD_recall: {test_pod}, '
                f'Test mTAR_precision: {test_tar}, '
                f'Test mKappa: {test_kappa}')


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
    if cuda:
        model.to(device=device)

    # train
    if args.test:
        test_net(args=args, cfg_oc=cfg_oc, model=model, device=device)
    else:
        try:
            train_net(args=args,
                      cfg_oc=cfg_oc,
                      model=model,
                      device=device)
        except KeyboardInterrupt:
            torch.save(model.state_dict(), os.path.join(dir_ckpt, 'INTERRUPTED.pth'))
            logger.info('Saved interrupt')
            raise
    experiment.finish()
