"""
One part box size prediction given box rotation.

Input: Image + one part mask + one part rotation
Output: one part size

"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_3d.ops.cd.chamfer import chamfer_distance
from common.nn.mlp import SharedMLP, MLP

from CompNet.models.backbones.resnet import resnet18
from CompNet.utils.obb_utils import get_box_pts_torch


def get_pts_from_size_rot(box):
    box_size = box[:, 3:6].contiguous()
    box_xdir = box[:, 6:9].contiguous()
    box_ydir = box[:, 9:12].contiguous()
    box_pts = get_box_pts_torch(size=box_size,
                                rx=box_xdir,
                                ry=box_ydir)
    return box_pts


def norm_size(box_size):
    """ Unit length size
    Args:
        box_size: torch.tensor, (batch_size, 3)
    """
    box_size = F.normalize(box_size, p=2, dim=1)
    return box_size


def get_zdir(xdir, ydir):
    """
    Args:
        xdir: torch.tensor, (batch_size, 3)
        ydir: torch.tensor, (batch_size, 3)
    """
    zdir = torch.cross(xdir, ydir, dim=1)
    zdir = F.normalize(zdir, dim=1, p=2)

    return zdir


def get_box_input(box1):
    """
    Args:
        box1: torch.tensor, (batch_size, 12), [center, size, rx, ry]

    Return:
        box1_input: torch.tensor, (batch_size, npts, 3)
        box1_size, torch.tensor, (batch_size, npts)
        box1_f: torch.tensor, (batch_size, C)

    """
    batch_size, _ = box1.shape
    box1_size, box1_xdir, box1_ydir = box1[:, 3:6], box1[:, 6:9], box1[:, 9:12]
    box1_zdir = get_zdir(box1_xdir, box1_ydir)
    box1_input = torch.stack([box1_xdir, box1_ydir, box1_zdir], dim=1)  # batch_size, npts, 3

    return box1_input


class AxisLengthNet(nn.Module):

    def __init__(self):
        super(AxisLengthNet, self).__init__()
        self.net_option = 'Size'
        # infer relationship of touch parts from [image, mask1, mask2]
        self.img_conv = nn.Sequential(
            resnet18(pretrained=False, input_channel=4, fc=False),
            nn.Conv2d(512, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4),
        )

        if self.net_option == 'Size':
            local_channels = (256, 256, 512)
            self.mlp_local = SharedMLP(256 + 3,
                                       local_channels)
            pts_in_channels = 3 + sum(local_channels) + local_channels[-1]
            self.dec_size = nn.Sequential(
                SharedMLP(pts_in_channels, (256, 256)),
                nn.Conv1d(256, 1, kernel_size=1, bias=True),
            )
        elif self.net_option == 'Scale':
            print(f'Add scale mode')
            exit()
        else:
            raise ValueError(f'Not supported TPBoxNet mode {self.net_option}')

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
        img = data_batch['img']  # B, C, H, W
        masks = data_batch['mask']  # (B, num_parts, H, W)
        boxes = data_batch['boxes']  # (B, num_parts, 12)
        batch_size, num_part, height, width = masks.shape

        img_expand = img.unsqueeze(dim=1).expand(-1, num_part, -1, -1, -1)  # B, 1, C, H, W
        masks_expand = masks.unsqueeze(dim=2)  # B, num_parts, 1, H, W

        img_expand = img_expand.contiguous()
        img1 = img_expand.view(batch_size * num_part, -1, height, width)
        mask1 = masks_expand.view(batch_size * num_part, -1, height, width)
        box1 = boxes.view(batch_size*num_part, -1)

        x1 = torch.cat([img1, mask1], dim=1)
        x1 = self.img_conv(x1)  # (batch_size, 256 , 1, 1)
        x1 = x1.view(batch_size*num_part, 256, 1)

        if self.net_option == 'Size':
            npts = 3
            box1_input = get_box_input(box1)  # (batch_size, npts, C)

            # image
            x1 = x1.expand(-1, -1, npts)  # (batch_size, C, npts)

            # estimate box1 size
            f1 = torch.cat([box1_input.permute(0, 2, 1), x1], dim=1)
            pred_box1_size = self.point_seg_size_module(f1,
                                                        box1_input)

            pred_pts = get_box_pts_torch(size=pred_box1_size,
                                         rx=box1[:, 6:9],
                                         ry=box1[:, 9:])

            preds = {'size': pred_box1_size,
                     'pts': pred_pts}

        else:
            raise ValueError(f'Not supported TPBoxNet mode {self.net_option}')

        return preds


class AxisLengthNetLoss(nn.Module):
    def __init__(self, mode='train', choice='L2'):
        super(AxisLengthNetLoss, self).__init__()
        self.mode = mode
        self.choice = choice
        logger.info(f'AxisLengthNetLoss use {self.choice} loss at {self.mode} stage.')

    def forward(self, preds, labels):
        losses = {}

        gt_box1 = labels['gt_box']  # batch_size, num_parts, 12
        batch_size, num_part, _ = gt_box1.shape
        gt_box1 = gt_box1.view(batch_size * num_part, -1)
        gt_box1_size = gt_box1[:, 3:6]

        pred_box1_size = preds['size']
        l2_size = torch.mean((gt_box1_size - pred_box1_size) ** 2)
        if self.mode == 'test':
            l2_size = l2_size.unsqueeze(0)
        losses['loss-l2'] = l2_size

        return losses


class AxisLengthNetMetric(nn.Module):
    def __init__(self, mode='train', choice='ChamferDist'):
        super(AxisLengthNetMetric, self).__init__()
        self.mode = mode
        self.choice = choice
        self.chamfer_dist = chamfer_distance
        logger.info(f'AxisLengthNetMetric use {self.choice} metric at {self.mode} stage.')

    def forward(self, preds, labels):
        metrics = {}
        if self.choice == 'ChamferDist':
            gt_box1 = labels['gt_box']  # batch_size, num_parts, 12
            batch_size, num_part, _ = gt_box1.shape
            gt_box1 = gt_box1.view(batch_size * num_part, -1)
            gt_size, gt_xydir = gt_box1[:, 3:6], gt_box1[:, 6:]

            gt_pts = get_box_pts_torch(size=gt_size,
                                       rx=gt_xydir[:, :3],
                                       ry=gt_xydir[:, 3:])
            pred_pts = preds['pts']

            dist1, dist2 = self.chamfer_dist(gt_pts, pred_pts)

            cd_dist = torch.mean(dist1 + dist2)
            if self.mode == 'test':
                cd_dist = cd_dist.unsqueeze(0)

            metrics['me-cd'] = cd_dist
        else:
            raise ValueError(f'Not supported metric {self.choice}')

        # size distance
        gt_box1 = labels['gt_box']  # batch_size, num_parts, 12
        batch_size, num_part, _ = gt_box1.shape
        gt_box1 = gt_box1.view(batch_size * num_part, -1)
        gt_box1_size = gt_box1[:, 3:6]
        pred_box1_size = preds['size']
        l1_size = torch.mean(torch.abs(gt_box1_size - pred_box1_size))
        if self.mode == 'test':
            metrics['me-l1size'] = torch.tensor([l1_size])
        else:
            metrics['me-l1size'] = l1_size

        return metrics


def build_AxisLengthNet(cfg, mode):
    net = AxisLengthNet()
    loss_fn = AxisLengthNetLoss(mode=mode, choice=cfg.MODEL.AxisLengthNet.LOSS,)
    metric_fn = AxisLengthNetMetric(mode=mode, choice=cfg.MODEL.AxisLengthNet.METRIC)

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    batch_size, num_part, height, width = 4, 8, 256, 256
    img = torch.randn(batch_size, 3, height, width).cuda()
    mask = torch.randn(batch_size, num_part, height, width).cuda()
    box = torch.randn(batch_size, num_part, 12).cuda()

    data_batch = {
        'img': img,
        'mask': mask,
        'gt_box': box,
        'boxes': box,
    }

    model = AxisLengthNet().cuda()

    preds = model(data_batch)
    for key, value in preds.items():
        logger.info(f'Preds {key}, {value.shape}')

    model_loss = AxisLengthNetLoss(choice='L2R').cuda()
    loss = model_loss(preds, data_batch)
    for key, value in loss.items():
        logger.info(f'Loss {key}, {value}')

    model_metric = AxisLengthNetMetric(choice='ChamferDist').cuda()
    metric = model_metric(preds, data_batch)
    for key, value in metric.items():
        logger.info(f'Loss {key}, {value}')


