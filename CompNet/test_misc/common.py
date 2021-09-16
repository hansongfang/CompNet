import numpy as np
from pathlib import Path

import torch

from CompNet.utils.obb_utils import match_xydir
from CompNet.utils.vis_utils import convert_box_to_ply, get_n_colors
from CompNet.utils.io_utils import write_img

ncanvas_template_file = './CompNet/viewer/templates/ply_ncanvas.html'
table_template_path = './CompNet/viewer/templates/table.html'

binary_center_model = ['JointNet']
unary_size_model = ['AxisLengthNet']
unary_rot_model = ['RotNet']


def logger_gt(t, part_list, output_dir, alpha=0.6):
    """Export ground truth visualizing material, including gt shape, masks and image"""
    gt_box_list, box_color_list = [], []
    for pid in part_list:
        gt_box_list.append(t.nodes[pid]['gt_box'])
        box_color_list.append(t.nodes[pid]['color'])

    gt_ply_file = str(Path(output_dir)/'gt.ply')
    convert_box_to_ply(gt_box_list,
                       box_color_list,
                       gt_ply_file)

    img = t.img
    write_img(str(Path(output_dir)/'img.jpg'), img)

    all_mask = np.zeros(img.shape, dtype=np.float)
    for pid in part_list:
        p_mask = t.nodes[pid]['mask_img']
        p_color = t.nodes[pid]['color']
        all_mask += p_mask[:, :, np.newaxis] * p_color
        all_mask = np.clip(all_mask, 0.0, 1.0)
    mask_img = alpha * img + (1 - alpha) * all_mask
    write_img(str(Path(output_dir)/'mask_img.jpg'), mask_img)


def match_gt_size(pred_box, gt_box):
    """
    Find corresponding gt obb axis size for pred box.
    Args:
        pred_box:
        gt_box:

    Returns:
        pred_size: pred box axis size
    """
    pred_xydir, gt_size, gt_xydir = pred_box[6:], gt_box[3:6], gt_box[6:]

    pred_xydir = torch.tensor(pred_xydir, dtype=torch.float).unsqueeze(0)
    gt_xydir = torch.tensor(gt_xydir, dtype=torch.float).unsqueeze(0)
    gt_size = torch.tensor(gt_size).float().unsqueeze(0)

    nn_inds = match_xydir(pred_xydir, gt_xydir)  # (b, 3)
    pred_size = torch.gather(gt_size, 1, nn_inds)
    pred_size = pred_size.squeeze(0).cpu().numpy()

    return pred_size


def get_loss_metric_key(cfg):
    if cfg.MODEL.CHOICE == 'JointNet':
        loss_key = 'loss-cd'
        metric_key = 'me-cd'
    elif cfg.MODEL.CHOICE == 'AxisLengthNet':
        loss_key = 'loss-l2'
        metric_key = 'me-cd'
    elif cfg.MODEL.CHOICE == 'RotNet':
        loss_key = 'loss-prior-emd'
        metric_key = 'me-rotgtsize'
    else:
        raise NotImplementedError()

    return loss_key, metric_key


def test_model(data_batch,
               model,
               loss_fn,
               metric_fn,
               loss_key,
               metric_key):
    """
    Args:
        data_batch: dict of model input

    Returns:
        loss: np.array, (batch_size, )

    """
    batch_size = data_batch['img'].shape[0]

    model.eval()
    metric_fn.eval()

    with torch.no_grad():
        preds = model(data_batch)
        loss_dict = loss_fn(preds, data_batch)
        metric_dict = metric_fn(preds, data_batch)

        losses = loss_dict[loss_key].detach().cpu().numpy()
        metrics = metric_dict[metric_key].detach().cpu().numpy()

    return preds, losses, metrics

