import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from CompNet.models.backbones.resnet import resnet18
from common_3d.ops.cd.chamfer import chamfer_distance
from common.nn.mlp import SharedMLP
from CompNet.utils.obb_utils import get_box_pts_torch


def get_box_input(box1):
    box1_size, box1_xdir, box1_ydir = box1[:, 3:6], box1[:, 6:9], box1[:, 9:12]
    box1_zdir = torch.cross(box1_xdir, box1_ydir, dim=1)
    box1_zdir = F.normalize(box1_zdir, dim=1, p=2)

    box1_input = torch.stack([box1_xdir, box1_ydir, box1_zdir], dim=1)  # batch_size, npts, 3
    return box1_input


class GroupAxisLengthNet(nn.Module):

    def __init__(self, mode='train'):
        super(GroupAxisLengthNet, self).__init__()
        self.mode = mode
        self.img_conv = nn.Sequential(
            resnet18(pretrained=False, input_channel=4, fc=False),
            nn.Conv2d(512, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4),
        )

        local_channels = (256, 256, 512)
        self.mlp_local = SharedMLP(256 + 3,
                                   local_channels)
        pts_in_channels = 3 + sum(local_channels) + local_channels[-1]
        self.dec_size = nn.Sequential(
            SharedMLP(pts_in_channels, (256, 256)),
            nn.Conv1d(256, 1, kernel_size=1, bias=True),
        )

    def point_seg_size_module(self, x, box_input):
        """ Point segmentation module to estimate size given box rotation

        Args:
            x: torch.tensor, (batch_size, C, npts)
            box_input: torch.tensor, (batch_size, npts, 3)

        Returns:
            pred_size: torch.tensor, (batch_size, 1, npts)

        """
        batch_size, _, npts = x.shape
        local_features = []
        for ind, mlp in enumerate(self.mlp_local):
            x = mlp(x)
            local_features.append(x)
        global_feature, max_indices = torch.max(x, 2)
        global_feature_expand = global_feature.unsqueeze(2).expand(-1, -1, npts)
        seg_features = [box_input.permute(0, 2, 1)] + local_features + [global_feature_expand]
        x = torch.cat(seg_features, dim=1)
        pred_size = self.dec_size(x)
        pred_size = pred_size.squeeze(dim=1)

        return pred_size

    def forward(self, data_batch):
        if self.mode in ['test']:
            imgs = data_batch['img']
            batch_size, num_part, _, height, width = imgs.shape
            masks = data_batch['mask']  # (B, num_part, H, W)
            masks = masks.unsqueeze(dim=2)  # (b, num_part, 1, H, W)

            x = torch.cat([imgs, masks], dim=2)
            x = x.view(batch_size * num_part, 4, height, width)
            x = self.img_conv(x)  # (batch_size * num_part, 256, 1, 1)
            x = x.view(batch_size, num_part, -1, 1)
            x12, _ = torch.max(x, dim=1)  # (batch_size, 256, 1)
        elif self.mode in ['train', 'val']:
            # training support only two
            # testing should support either number of equal parts
            imgs = data_batch['img']
            batch_size, num_part, _, height, width = imgs.shape
            img1 = data_batch['img'][:, 0, :, :, :]  # (B, C, H, W)
            img2 = data_batch['img'][:, 1, :, :, :]  # (B, C, H, W)
            # batch_size, _, height, width = img1.shape

            mask1 = data_batch['mask'][:, 0, :, :]  # (B, H, W)
            mask2 = data_batch['mask'][:, 1, :, :]  # (B, H, W)

            x1 = torch.cat([img1, mask1.unsqueeze(dim=1)], dim=1)
            x2 = torch.cat([img2, mask2.unsqueeze(dim=1)], dim=1)

            x1 = self.img_conv(x1)  # (batch_size, 256 , 1, 1)
            x2 = self.img_conv(x2)

            x1 = x1.view(batch_size, 256, 1)  # batch_size, 256, 1
            x2 = x2.view(batch_size, 256, 1)  # batch_size, 256, 1

            x12 = torch.stack([x1, x2], dim=1)
            x12, _ = torch.max(x12, dim=1)  # merge pair part feature
        else:
            raise NotImplementedError()

        # using box1 axis only
        box1 = data_batch['box'][:, 0, :]  # (B, 12)

        npts = 3
        box1_input = get_box_input(box1)  # (batch_size, npts, C)
        x12 = x12.expand(-1, -1, npts)  # (batch_size, C, npts)

        # estimate box1 size
        f1 = torch.cat([box1_input.permute(0, 2, 1), x12], dim=1)
        pred_box1_size = self.point_seg_size_module(f1,
                                                    box1_input)

        pred_pts = get_box_pts_torch(size=pred_box1_size,
                                     rx=box1[:, 6:9],
                                     ry=box1[:, 9:])

        pred_box1_size = pred_box1_size.unsqueeze(dim=1).expand(batch_size, num_part, 3)
        pred_box1_size = pred_box1_size.contiguous()

        preds = {'size': pred_box1_size,
                 'pts': pred_pts}

        return preds


class GroupAxisLengthNetLoss(nn.Module):
    def __init__(self, mode='train'):
        super(GroupAxisLengthNetLoss, self).__init__()
        self.mode = mode
        logger.info(f'GroupAxisLengthNetLoss loss at {self.mode} stage.')

    def forward(self, preds, labels):
        losses = {}

        gt_box1 = labels['gt_box'][:, 0, :]  # batch_size, num_parts, 12
        gt_box1_size = gt_box1[:, 3:6]

        pred_box1_size = preds['size'][:, 0, :]  # batch_size, num_parts, 3
        l2_size = torch.mean((gt_box1_size - pred_box1_size) ** 2)
        if self.mode == 'test':
            l2_size = l2_size.unsqueeze(0)
        losses['loss-l2'] = l2_size
        return losses


class GroupAxisLengthNetMetric(nn.Module):
    def __init__(self, mode='train', choice='ChamferDist'):
        super(GroupAxisLengthNetMetric, self).__init__()
        self.mode = mode
        self.choice = choice
        self.chamfer_dist = chamfer_distance
        logger.info(f'GroupAxisLengthNetMetric use {self.choice} metric at {self.mode} stage.')

    def forward(self, preds, labels):
        metrics = {}
        if self.choice == 'ChamferDist':
            gt_box1 = labels['gt_box'][:, 0, :]  # batch_size, 12
            batch_size, _ = gt_box1.shape
            gt_size, gt_xydir = gt_box1[:, 3:6], gt_box1[:, 6:]

            gt_pts = get_box_pts_torch(size=gt_size,
                                       rx=gt_xydir[:, :3],
                                       ry=gt_xydir[:, 3:])
            pred_pts = preds['pts']  # batch_size, 8, 3

            dist1, dist2 = self.chamfer_dist(gt_pts, pred_pts)

            cd_dist = torch.mean(dist1 + dist2)
            if self.mode == 'test':
                cd_dist = cd_dist.unsqueeze(0)

            metrics['me-cd'] = cd_dist
        else:
            raise ValueError(f'Not supported metric {self.choice}')

        if self.mode == 'test':
            # size distance
            gt_box1 = labels['gt_box']  # batch_size, num_parts, 12
            batch_size, num_part, _ = gt_box1.shape
            gt_box1 = gt_box1.view(batch_size * num_part, -1)
            gt_box1_size = gt_box1[:, 3:6]
            pred_box1_size = preds['size']
            pred_box1_size = pred_box1_size.view(batch_size*num_part, -1)
            l1_size = torch.mean(torch.abs(gt_box1_size - pred_box1_size))
            metrics['me-l1size'] = torch.tensor([l1_size])

        return metrics


def build_GroupAxisLengthNet(cfg, mode):
    net = GroupAxisLengthNet(mode=mode)
    loss_fn = GroupAxisLengthNetLoss(mode=mode)
    metric_fn = GroupAxisLengthNetMetric(mode=mode,
                                         choice=cfg.MODEL.GroupAxisLengthNet.METRIC)

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    batch_size, num_part, height, width = 4, 8, 256, 256
    img = torch.randn(batch_size, num_part, 3, height, width).cuda()
    mask = torch.randn(batch_size, num_part, height, width).cuda()
    box = torch.randn(batch_size, num_part, 12).cuda()

    data_batch = {
        'img': img,
        'mask': mask,
        'gt_box': box,
        'box': box,
    }

    model = GroupAxisLengthNet(mode='train').cuda()

    preds = model(data_batch)
    for key, value in preds.items():
        logger.info(f'Preds {key}, {value.shape}')

    model_loss = GroupAxisLengthNetLoss(choice='L2').cuda()
    loss = model_loss(preds, data_batch)
    for key, value in loss.items():
        logger.info(f'Loss {key}, {value}')

    model_metric = GroupAxisLengthNetMetric(choice='ChamferDist').cuda()
    metric = model_metric(preds, data_batch)
    for key, value in metric.items():
        logger.info(f'Loss {key}, {value}')













