"""
Pure image model to detect whether two part are adjacent or not.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from CompNet.models.backbones.resnet import resnet18


class SizeRelationNet(nn.Module):
    def __init__(self):
        super(SizeRelationNet, self).__init__()

        self.img_conv = nn.Sequential(
            resnet18(pretrained=False, input_channel=5, fc=False),
            nn.Conv2d(512, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4),
        )

        self.dec_adj = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, bias=True),
            nn.Sigmoid()  # OBB size in range (0, 1)
        )

    def forward(self, data_batch):
        img2 = data_batch['img'][:, 1, :, :, :]  # (B, C, H, W)
        batch_size, _, height, width = img2.shape

        mask1 = data_batch['mask'][:, 0, :, :]  # (B, H, W)
        mask2 = data_batch['mask'][:, 1, :, :]  # (B, H, W)

        x1 = torch.cat([img2, mask1.unsqueeze(dim=1), mask2.unsqueeze(dim=1)], dim=1)  # B, 5, H, W
        x1 = self.img_conv(x1)  # B, C, 1, 1
        pred_adj = self.dec_adj(x1)  # B, 1, 1, 1
        pred_adj = pred_adj.view(batch_size, 1)

        preds = {
            'adj': pred_adj,
        }
        return preds


class SizeRelationNetLoss(nn.Module):
    def __init__(self, mode='train'):
        super(SizeRelationNetLoss, self).__init__()
        # self.criterion = nn.BCELoss()
        self.mode = mode

    def forward(self, preds, labels):
        pred_adj = preds['adj']
        gt_adj = labels['adj']
        if 'weight' in labels:
            weight = labels['weight']
            loss = F.binary_cross_entropy(pred_adj, gt_adj, weight)
        else:
            loss = F.binary_cross_entropy(pred_adj, gt_adj)

        # loss = self.criterion(pred_adj, gt_adj)
        losses = {}
        if self.mode == 'test':
            losses['loss_bce'] = torch.tensor([loss])
        else:
            losses['loss_bce'] = loss

        return losses


class SizeRelationNetMetric(nn.Module):

    def __init__(self, mode='train'):
        super(SizeRelationNetMetric, self).__init__()
        self.mode = mode

    def forward(self, preds, labels):
        pred_adj = preds['adj']
        gt_adj = labels['adj']

        y_pred_tag = torch.round(pred_adj)
        pred_eq_gt = (y_pred_tag == gt_adj).float()

        # positive accurate
        if gt_adj.sum() > 0.0:
            pos_acc = (gt_adj * y_pred_tag).sum().float() / gt_adj.sum()
        else:
            pos_acc = 1.0

        if (1.0 - gt_adj).sum() > 0.0:
            false_acc = ((1.0 - gt_adj) * (1.0 - y_pred_tag)).sum().float() / (1.0 - gt_adj).sum()
        else:
            false_acc = 1.0

        correct_results_sum = (y_pred_tag == gt_adj).sum().float()
        acc = correct_results_sum / gt_adj.shape[0]

        metrics = {}
        if self.mode == 'test':
            metrics['me-acc'] = torch.tensor([acc])
            metrics['me-pred'] = pred_adj
        else:
            metrics['me-acc'] = acc
            metrics['me-positive-acc'] = pos_acc
            metrics['me-false-acc'] = false_acc

        return metrics


def build_SizeRelationNet(cfg, mode):
    net = SizeRelationNet()
    loss_fn = SizeRelationNetLoss(mode=mode)
    metric_fn = SizeRelationNetMetric(mode=mode)

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    num_part = 2
    batch_size, height, width = 4, 256, 256
    img = torch.randn(batch_size, num_part, 3, height, width).cuda()
    mask = torch.randn(batch_size, num_part, height, width).cuda()
    adj = torch.ones(batch_size, 1).cuda()

    data_batch = {
        'img': img,
        'mask': mask,
        'adj': adj,
    }

    model = SizeRelationNet().cuda()
    preds = model(data_batch)
    for key, value in preds.items():
        logger.info(f'Preds {key}, {value.shape}')

    model_loss = SizeRelationNetLoss().cuda()
    loss = model_loss(preds, data_batch)
    for key, value in loss.items():
        logger.info(f'Loss {key}, {value}')

    model_metric = SizeRelationNetMetric().cuda()
    metric = model_metric(preds, data_batch)
    for key, value in metric.items():
        logger.info(f'Loss {key}, {value}')




