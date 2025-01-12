import numpy as np
import torch
from torch.nn import functional as F
from torch import Tensor
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    # print(f'dice coeff input size: {input.shape}')
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        # print(inter.item(), sets_sum.item())
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
    # print(input.shape, target.shape)  # torch.Size([4, 1, 640, 1280]) torch.Size([4, 1, 640, 1280])
    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size(), f'input size: {input.size()} != target size: {target.size()}'
    # print(f'dice loss input size: {input.shape}')  # dice loss input size: torch.Size([4, 2, 640, 1280])
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def IOU_csi(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6, channel=1):
    # Average of iou for all batches, or for a single mask
    assert input.size() == target.size()
    # print(f'IoU input size: {input.size()}')
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f'IoU: asked to reduce batch but got tensor without batch dimension (input shape {input.shape})')
    if input.dim() == 2 or reduce_batch_first:
        channel = 1
        intersection = torch.dot(input.reshape(-1) / channel, target.reshape(-1) / channel)
        union = torch.sum(input).item() / channel + torch.sum(target).item() / channel
        if union == 0:
            union = 2 * intersection
        return (intersection + epsilon) / (union - intersection + epsilon)
    else:
        iou = 0
        for i in range(input.shape[0]):
            iou += IOU_csi(input[i, ...], target[i, ...], channel=channel)
        return iou / input.shape[0]


def mIOU_csi(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # = CSI
    assert input.size() == target.size()
    iou = 0
    for channel in range(input.shape[1]):
        iou += IOU_csi(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon,
                       channel=channel + 1)
    return iou / input.shape[1]


def POD_recall(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6, channel=1):
    # Average of POD(Recall) for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f'POD: asked to reduce batch but got tensor without batch dimension (input shape {input.shape})')
    if input.dim() == 2 or reduce_batch_first:
        channel = 1
        TP_intersection = torch.dot(input.reshape(-1) / channel, target.reshape(-1) / channel)
        union = torch.sum(input).item() / channel + torch.sum(target).item() / channel
        TP_FN = torch.sum(target).item() / channel
        if union == 0:
            TP_FN = TP_intersection
        return (TP_intersection + epsilon) / (TP_FN + epsilon)
    else:
        # batch mean
        pod = 0
        for i in range(input.shape[0]):
            pod += POD_recall(input[i, ...], target[i, ...], channel=channel)
        return pod / input.shape[0]


def mPOD_recall(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # = Recall
    assert input.size() == target.size()
    pod = 0
    for channel in range(input.shape[1]):
        pod += POD_recall(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon,
                          channel=channel + 1)
    return pod / input.shape[1]


def TAR_precision(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6, channel=1):
    # Average of TAR(Precision) for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f'TAR: asked to reduce batch but got tensor without batch dimension (input shape {input.shape})')
    if input.dim() == 2 or reduce_batch_first:
        channel = 1
        TP_intersection = torch.dot(input.reshape(-1) / channel, target.reshape(-1) / channel)
        union = torch.sum(input).item() / channel + torch.sum(target).item() / channel
        TP_FP = torch.sum(input).item() / channel
        if union == 0:
            TP_FP = TP_intersection
        return (TP_intersection + epsilon) / (TP_FP + epsilon)
    else:
        # batch mean
        tar = 0
        for i in range(input.shape[0]):
            tar += TAR_precision(input[i, ...], target[i, ...], channel=channel)
        return tar / input.shape[0]


def mTAR_precision(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # = Precision
    assert input.size() == target.size()
    tar = 0
    for channel in range(input.shape[1]):
        tar += TAR_precision(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon,
                             channel=channel + 1)
    return tar / input.shape[1]


def Kappa(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6, channel=1):
    """计算kappa值系数"""
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f'Kappa: asked to reduce batch but got tensor without batch dimension (input shape {input.shape})')
    if input.dim() == 2 or reduce_batch_first:
        channel = 1
        TP = torch.dot(input.reshape(-1) / channel, target.reshape(-1) / channel)
        TP_FP = (torch.sum(input).item() / channel)
        TP_FN = (torch.sum(target).item() / channel)
        FP = TP_FP - TP
        FN = TP_FN - TP
        TN = (input.shape[-2] * input.shape[-1] - TP - FP - FN)
        # union = torch.sum(input).item() / channel + torch.sum(target).item() / channel
        po = (TP + TN) / (TP + TN + FP + FN)
        pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (input.shape[-2] * input.shape[-1]) ** 2
        # if union == 0:
        #     TP_FP = TP
        return (po - pe + epsilon) / (1 - pe + epsilon)
    else:
        # batch mean
        kappa = 0
        for i in range(input.shape[0]):
            kappa += Kappa(input[i, ...], target[i, ...], channel=channel)
        return kappa / input.shape[0]


def mKappa(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    assert input.size() == target.size()
    kappa = 0
    for channel in range(input.shape[1]):
        kappa += Kappa(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon,
                       channel=channel + 1)
    return kappa / input.shape[1]


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, num_classes, patch_size=[256, 256], test_save_path=None, case=None,
                       z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, num_classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list
