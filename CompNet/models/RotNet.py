import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchgeometry as tgm
from CompNet.models.backbones.resnet import resnet18
from CompNet.utils.obb_utils import get_rotmat_from_xydir_v2
from common_3d.ops.cd.chamfer import chamfer_distance
from CompNet.models.rot_loss_util import rot_minn_emd_loss_v2, rot_match_size_loss


class RotNet(nn.Module):

    def __init__(self,):
        super(RotNet, self).__init__()

        self.img_conv = resnet18(pretrained=False,
                                 input_channel=4,
                                 fc=False)
        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4),
        )

        # Multi-brach prediction
        self.k = 4
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.k * (4 + 1), kernel_size=1, bias=True),
        )

    def forward(self, data_batch):
        img = data_batch['img']  # (B, C, H, W)
        masks = data_batch['mask']  # (B, num_parts, H, W)
        batch_size, num_part, height, width = masks.shape

        img_expand = img.unsqueeze(dim=1).expand(-1, num_part, -1, -1, -1)
        masks_expand = masks.unsqueeze(dim=2)
        x = torch.cat([img_expand, masks_expand], dim=2)
        x = x.view(batch_size * num_part, -1, height, width).contiguous()
        img_feature = self.img_conv(x)
        conv_feature = self.conv(img_feature)

        output = self.decoder(conv_feature).reshape(-1, 5)  # [B * NP * K, 4 + 1]
        box_quat, rot_logit = torch.split(output, [4, 1], dim=-1)
        box_quat = F.normalize(box_quat, p=2, dim=-1)
        box_angle_axis = tgm.core.quaternion_to_angle_axis(box_quat)  # [B * NP, 3]
        box_rot = tgm.core.angle_axis_to_rotation_matrix(box_angle_axis)[..., :3, :3]

        rot_logits = rot_logit.reshape(-1, self.k, 1)
        box_rots = box_rot.reshape(-1, self.k, 3, 3)

        best_index = rot_logits.argmax(1, keepdim=True)  # [B * NP, 1, 1]
        best_index_expand = best_index.unsqueeze(-1).expand(-1, 1, 3, 3)
        box_rot_best = torch.gather(box_rots, dim=1, index=best_index_expand).squeeze(1)  # [B * NP, 3, 3]
        box_xydir = torch.cat([box_rot_best[:, :, 0], box_rot_best[:, :, 1]], dim=-1)

        preds = {'xydir': box_xydir,
                 'rot': box_rot_best,
                 'rots': box_rots,
                 'rot_logits': rot_logits}

        return preds


class RotNetLoss(nn.Module):
    def __init__(self,
                 choice='MixMinNEMD-0.1',
                 mode='train'):
        super(RotNetLoss, self).__init__()
        self.choice = choice
        self.mode = mode
        print(f'Loss choice: {self.choice} in mode {self.mode}')

    def forward(self, preds, labels):
        gt_box = labels['gt_box']
        batch_size, num_parts, _ = gt_box.shape
        gt_box = gt_box.view(batch_size * num_parts, -1)

        losses = {}
        sigma = float(self.choice.split('-')[1])
        beta = 1.0

        pred_box_rots = preds['rots']  # [B * NP, K, 3, 3]
        k = pred_box_rots.size(1)

        gt_box_size, gt_box_xydir = gt_box[:, 3:6], gt_box[:, 6:]
        gt_box_rot = get_rotmat_from_xydir_v2(gt_box_xydir, orthonormal=True)
        gt_box_size_repeat = torch.repeat_interleave(gt_box_size, k, dim=0)
        gt_box_rot_repeat = torch.repeat_interleave(gt_box_rot, k, dim=0)

        emd_dist = rot_minn_emd_loss_v2(gt_box_size=gt_box_size_repeat,
                                        gt_box_rot=gt_box_rot_repeat,
                                        pred_box_rot=pred_box_rots.reshape(-1, 3, 3),
                                        relative_size=False,
                                        reduce=False,
                                        )
        emd_dist = emd_dist.reshape(-1, k)
        emd_dist_min, min_idx = emd_dist.min(dim=1)  # [B * NP]

        # mixture of experts
        rot_logits = preds['rot_logits'].squeeze(-1)  # [B * NP, K]
        rot_probs = F.softmax(rot_logits, dim=-1)  # [B * NP, K]
        rot_log_probs = torch.log(rot_probs.clamp(min=1e-6))
        log_exp = rot_log_probs - emd_dist / sigma
        nll = - torch.logsumexp(log_exp, dim=-1)
        nll = nll * beta

        if self.mode != 'test':
            emd_dist_min = emd_dist_min.mean()
            nll = nll.mean()
            losses['loss-prior-emd'] = emd_dist_min
            losses['loss-nll'] = nll
        else:
            losses['loss-prior-emd'] = emd_dist_min

        return losses


class RotNetMetric(nn.Module):
    def __init__(self,
                 choice='Rot',
                 relative_size=False,
                 mode='train'):
        super(RotNetMetric, self).__init__()
        self.chamfer_dist = chamfer_distance
        self.choice = choice
        self.relative_size = relative_size
        self.mode = mode
        print(f'Metric choice: {self.choice} in mode {self.mode}')

    def forward(self, preds, labels):
        gt_box = labels['gt_box']
        batch_size, num_parts, _ = gt_box.shape
        gt_box = gt_box.view(batch_size * num_parts, -1)
        gt_size, gt_xdir, gt_ydir = gt_box[:, 3:6], gt_box[:, 6:9], gt_box[:, 9:12]

        pred_xydir = preds['xydir']
        gt_xydir = gt_box[:, 6:]
        cd_dist = rot_match_size_loss(pred_xydir,
                                      gt_size,
                                      gt_xydir,
                                      cd_func=self.chamfer_dist)
        if self.mode == 'test':
            cd_dist = cd_dist.unsqueeze(0)  # test evaluate parts of a shape

        metrics = {'me-rotgtsize': cd_dist}

        return metrics


def build_RotNet(cfg, mode):
    if mode == 'test':
        assert cfg.TEST.BATCH_SIZE == 1  # check loss and metric in test mode

    net = RotNet()

    loss_fn = RotNetLoss(choice=cfg.MODEL.ROTNET.LOSS, mode=mode)

    metric_fn = RotNetMetric(choice=cfg.MODEL.ROTNET.METRIC, mode=mode)

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    batch_size, num_part, height, width = 2, 3, 256, 256
    img = torch.randn(batch_size, 3, height, width).cuda()
    mask = torch.randn(batch_size, num_part, height, width).cuda()
    box = torch.randn(batch_size, num_part, 12).cuda()

    data_batch = {
        'img': img,
        'mask': mask,
        'gt_box': box
    }

    model = RotNet().cuda()

    preds = model(data_batch)
    model_loss = RotNetLoss(choice='MixMinNEMD').cuda()
    # model_loss = RotNetLoss(choice='MSE').cuda()
    loss = model_loss(preds, data_batch)
    print(loss)
    model_metric = RotNetMetric(choice='RotGTSize').cuda()
    metric = model_metric(preds, data_batch)
    print(metric)
