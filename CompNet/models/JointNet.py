import os.path as osp
import sys

sys.path.insert(0, osp.dirname(__file__) + '/..')

from loguru import logger

import torch
import torch.nn as nn

from CompNet.models.backbones.resnet import resnet18
from common_3d.ops.cd.chamfer import chamfer_distance
from common.nn.mlp import SharedMLP
from CompNet.utils.obb_utils import get_box_pts_torch


def get_pts_from_size_rot(box):
    box_size = box[:, 3:6].contiguous()
    box_xdir = box[:, 6:9].contiguous()
    box_ydir = box[:, 9:12].contiguous()
    box_pts = get_box_pts_torch(size=box_size,
                                rx=box_xdir,
                                ry=box_ydir)
    return box_pts


class JointNet(nn.Module):

    def __init__(self, norm_scale=False, mask_img=False):
        super(JointNet, self).__init__()

        self.img_conv = nn.Sequential(
            resnet18(pretrained=False, input_channel=5, fc=False),
            nn.Conv2d(512, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4),
        )

        local_channels = (256, 256, 512)
        self.mlp_local = SharedMLP(512+3,
                                   (256, 256, 512))
        pts_in_channels = 3 + sum(local_channels) + local_channels[-1]

        self.dec_w = nn.Sequential(
            SharedMLP(pts_in_channels, (256, 256)),
            nn.Conv1d(256, 1, kernel_size=1, bias=True),
            nn.Softmax(dim=2)
        )
        self.norm_scale = norm_scale
        self.mask_img = mask_img
        print(f'JointNet norm_scale input box : {self.norm_scale}')
        print(f'JointNet mask_img: {self.mask_img}')

    def forward(self, data_batch):
        img1 = data_batch['img'][:, 0, :, :, :]  # (B, C, H, W)
        img2 = data_batch['img'][:, 1, :, :, :]  # (B, C, H, W)
        batch_size, _, height, width = img1.shape

        mask1 = data_batch['mask'][:, 0, :, :]  # (B, H, W)
        mask2 = data_batch['mask'][:, 1, :, :]  # (B, H, W)

        box1 = data_batch['box'][:, 0, :]  # (B, 12)
        box2 = data_batch['box'][:, 1, :]  # (B, 12)

        if self.norm_scale:
            box1_size = box1[:, 3:6]
            box2_size = box2[:, 3:6]
            scale1, _ = torch.max(box1_size, dim=1)
            scale2, _ = torch.max(box2_size, dim=1)
            scale, _ = torch.max(torch.stack((scale1, scale2)), dim=0)
            assert len(scale.shape) == 1
            # print(f'Norm scale box: {scale}')
            # exit()

        x1 = torch.cat([img1, mask1.unsqueeze(dim=1), mask2.unsqueeze(dim=1)], dim=1)
        x2 = torch.cat([img2, mask2.unsqueeze(dim=1), mask1.unsqueeze(dim=1)], dim=1)

        x1 = self.img_conv(x1)  # (batch_size, 256 , 1, 1)
        x2 = self.img_conv(x2)
        # print(f'Extracted feature', x1.shape)
        # exit()

        x1 = x1.view(batch_size, 256, 1).expand(-1, -1, 8)
        x2 = x2.view(batch_size, 256, 1).expand(-1, -1, 8)

        box1_pts = get_pts_from_size_rot(box1)  # (batch_size, 8, 3)
        box2_pts = get_pts_from_size_rot(box2)  # (batch_size, 8, 3)
        if self.norm_scale:
            box1_pts = box1_pts / scale.view(batch_size, 1, 1)
            box2_pts = box2_pts / scale.view(batch_size, 1, 1)
        f1 = torch.cat([x1, x2, box1_pts.permute(0, 2, 1)], dim=1)
        local_features = []
        x = f1
        for ind, mlp in enumerate(self.mlp_local):
            x = mlp(x)
            local_features.append(x)
        global_feature, max_indices = torch.max(x, 2)
        global_feature_expand = global_feature.unsqueeze(2).expand(-1, -1, 8)
        seg_features = [box1_pts.permute(0, 2, 1)] + local_features + [global_feature_expand]
        x = torch.cat(seg_features, dim=1)
        w1 = self.dec_w(x)
        w1 = w1.permute(0, 2, 1).contiguous()

        f2 = torch.cat([x2, x1, box2_pts.permute(0, 2, 1)], dim=1)
        local_features = []
        x = f2
        for ind, mlp in enumerate(self.mlp_local):
            x = mlp(x)
            local_features.append(x)
        global_feature, max_indices = torch.max(x, 2)
        global_feature_expand = global_feature.unsqueeze(2).expand(-1, -1, 8)
        seg_features = [box2_pts.permute(0, 2, 1)] + local_features + [global_feature_expand]
        x = torch.cat(seg_features, dim=1)
        w2 = self.dec_w(x)
        w2 = w2.permute(0, 2, 1).contiguous()

        touch_pts1 = torch.sum(w1 * box1_pts, dim=1)  # in box1 coordinate
        touch_pts2 = torch.sum(w2 * box2_pts, dim=1)  # in box2 coordinate

        if self.norm_scale:
            box1_pts = box1_pts * scale.view(batch_size, 1, 1)
            box2_pts = box2_pts * scale.view(batch_size, 1, 1)
            touch_pts1 = touch_pts1 * scale.view(batch_size, 1)
            touch_pts2 = touch_pts2 * scale.view(batch_size, 1)

        # box1_loc + touch_pts1 = box2_loc + touch_pts
        box1_loc = box1[:, :3]  # (batch_size, 3)
        box2_loc = touch_pts1 - touch_pts2 + box1_loc  # (batch_size, 3)
        pred_pts = box2_pts + box2_loc.unsqueeze(dim=1)

        preds = {'center': box2_loc,
                 'pts': pred_pts,
                 'offset': touch_pts1 - touch_pts2}
        return preds


class JointNetLoss(nn.Module):
    """Supporting batch chamfer loss when testing with mode."""
    def __init__(self, option='ChamferDist', mode='train'):
        super(JointNetLoss, self).__init__()
        # self.chamfer_dist = ChamferDistance()
        self.chamfer_dist = chamfer_distance
        self.option = option
        self.mode = mode
        logger.info(f'JointNetLoss use {self.option} loss at {self.mode} stage.')

    def forward(self, preds, labels):
        # evaluate box2
        gt_box = labels['gt_box'][:, 1, :]  # (B, 12) -> find the relative position, how to combine together

        gt_center = gt_box[:, :3]
        gt_size = gt_box[:, 3:6]
        gt_xdir = gt_box[:, 6:9]
        gt_ydir = gt_box[:, 9:12]
        gt_pts = get_box_pts_torch(size=gt_size,
                                   rx=gt_xdir,
                                   ry=gt_ydir,
                                   center=gt_center)
        losses = {}
        if self.option == 'ChamferDist':
            # Chamfer distance
            pred_pts = preds['pts']
            dist1, dist2 = self.chamfer_dist(gt_pts, pred_pts)
            assert dist1.shape[1] == 8
            assert dist2.shape[1] == 8
            if self.mode == 'test':
                assert len(dist1.shape) == 2
                cd_dist = torch.mean(dist1+dist2, dim=-1)
            else:
                cd_dist = torch.mean(dist1 + dist2)
            losses['loss-cd'] = cd_dist
        elif self.option == 'L1':
            pred_center = preds['center']
            loss_center = torch.mean(torch.abs(gt_center - pred_center)) * 3.0
            losses['loss-l1center'] = loss_center
        elif self.option == 'L2':
            pred_center = preds['center']
            loss_center = torch.mean((gt_center - pred_center)**2)
            losses['loss-l2center'] = loss_center
        elif self.option == 'CD_Sqrt':
            pred_pts = preds['pts']
            dist1, dist2 = self.chamfer_dist(gt_pts, pred_pts)
            dist1 = torch.sqrt(dist1)
            dist2 = torch.sqrt(dist2)
            cd_dist = torch.mean(dist1 + dist2)
            losses['loss-cd2'] = cd_dist
        elif self.option == 'Mix':
            pred_center = preds['center']
            loss_center = torch.mean(torch.abs(gt_center - pred_center)) * 3.0
            losses['loss-center'] = loss_center

            pred_pts = preds['pts']
            dist1, dist2 = self.chamfer_dist(gt_pts, pred_pts)
            dist1 = torch.sqrt(dist1)
            dist2 = torch.sqrt(dist2)
            cd_dist = torch.mean(dist1 + dist2)
            losses['loss-cd2'] = cd_dist

        return losses


class JointNetMetric(nn.Module):
    def __init__(self, option='ChamferDist', mode='train'):
        super(JointNetMetric, self).__init__()
        # self.chamfer_dist = ChamferDistance()
        self.chamfer_dist = chamfer_distance
        self.option = option
        self.mode = mode
        logger.info(f'JointNetMetric use {self.option} metric at {self.mode} stage.')

    def forward(self, preds, labels):
        metrics = {}
        gt_box = labels['gt_box'][:, 1, :]  # (B, 12) -> find the relative position, how to combine together
        gt_center, gt_size, gt_xydir = gt_box[:, :3], gt_box[:, 3:6], gt_box[:, 6:]

        pred_center = preds['center']  # (B, 3)
        l1_center = torch.mean(torch.abs(gt_center - pred_center))
        if self.mode == 'test':
            metrics['me-l1center'] = torch.tensor([l1_center])
        else:
            metrics['me-l1center'] = l1_center

        l2_center = torch.mean((gt_center - pred_center) ** 2)
        if self.mode == 'test':
            metrics['me-l2center'] = torch.tensor([l2_center])
        else:
            metrics['me-l2center'] = l2_center

        gt_pts = get_box_pts_torch(size=gt_size,
                                   rx=gt_xydir[:, :3],
                                   ry=gt_xydir[:, 3:],
                                   center=gt_center)
        pred_pts = preds['pts']
        dist1, dist2 = self.chamfer_dist(gt_pts, pred_pts)

        if self.mode == 'test':
            cd_dist = torch.mean(dist1 + dist2, dim=-1)
        else:
            cd_dist = torch.mean(dist1 + dist2)

        metrics['me-cd'] = cd_dist
        return metrics


def build_JointNet(cfg, mode):
    net = JointNet(norm_scale=cfg.MODEL.JOINTNET.NORM_SCALE,
                   mask_img=cfg.MODEL.JOINTNET.MASK_IMAGE)

    loss_fn = JointNetLoss(option=cfg.MODEL.JOINTNET.LOSS, mode=mode)

    metric_fn = JointNetMetric(option=cfg.MODEL.JOINTNET.METRIC, mode=mode)

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    num_part = 2
    batch_size, height, width = 4, 256, 256
    img = torch.randn(batch_size, num_part, 3, height, width).cuda()
    mask = torch.randn(batch_size, num_part, height, width).cuda()
    box = torch.randn(batch_size, num_part, 12).cuda()
    gt_box = torch.tensor(box)

    data_batch = {
        'img': img,
        'mask': mask,
        'box': box,
        'gt_box': gt_box,
    }

    model = JointNet().cuda()

    preds = model(data_batch)
    preds_center = preds['center']
    preds_pts = preds['pts']

    print('center: ', preds_center.shape)
    print('pts: ', preds_pts.shape)

    model_loss = JointNetLoss().cuda()
    loss = model_loss(preds, data_batch)
    print(loss)

    model_metric = JointNetMetric().cuda()
    metric = model_metric(preds, data_batch)
    for key, value in metric.items():
        logger.info(f'Loss {key}, {value}')







