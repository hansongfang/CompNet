"""

"""
from pathlib import Path
import numpy as np
np.set_printoptions(precision=2, suppress=True)
from loguru import logger
import networkx as nx

import torch

from CompNet.test_misc.shape_graph import ShapeGraph
from CompNet.test_misc.common import match_gt_size, logger_gt, test_model
from CompNet.utils.vis_utils import convert_box_to_ply, get_n_colors
from CompNet.viewer.tool import get_ncanvas_html
from CompNet.test_misc.common import ncanvas_template_file
from CompNet.test_misc.common import unary_rot_model, unary_size_model
from CompNet.test_misc.common import get_loss_metric_key


def get_parts_batch(t, pid_list=[]):
    """
     Get shape graph parts data.

    Args:
        t: Shape Graph

    """
    img = t.img

    if len(pid_list) == 0:
        pid_list = list(t.nodes)

    parts_mask = []
    parts_box = []
    gt_boxes = []
    parts_id = []
#     print(f'There are {len(t.nodes)} nodes')
    for pid in pid_list:
        mask = t.nodes[pid]['mask_img']
        box = t.nodes[pid]['pred_box']
        logger.debug(f'Loading part {pid} box, {box}')
        gt_box = t.nodes[pid]['gt_box']
        parts_mask.append(mask)
        parts_box.append(box)
        gt_boxes.append(gt_box)
        parts_id.append(pid)
#         print(f'Part {pid}', mask.shape)
        
    parts_mask = np.array(parts_mask)
    parts_box = np.array(parts_box)
    gt_boxes = np.array(gt_boxes)
    parts_id = np.array(parts_id)

    img = torch.tensor(img.copy()).permute(2, 0, 1).float().unsqueeze(0)  # batch_size, 3, height, width
    parts_mask = torch.tensor(parts_mask).float().unsqueeze(0)  # batch_size, num_part, height, width
    parts_box = torch.tensor(parts_box).float().unsqueeze(0)  # batch_size, num_part, 12
    gt_boxes = torch.tensor(gt_boxes).float().unsqueeze(0)  # batch_size, num_part, 12
    parts_id = torch.tensor(parts_id).int().unsqueeze(0)  # batch_size, num_parts

    return {
        'img': img,
        'obj_name': t.name,
        'pids': parts_id,
        'mask': parts_mask,
        'boxes': parts_box,
        'gt_box': gt_boxes
    }


def logger_pred_unary(t,
                      part_list,
                      pred_ply_fn,
                      use_gt_rot=False,
                      use_gt_size=False,
                      use_gt_center=False):
    """Input is image and one mask, output one part"""
    pred_box_list, box_color_list = [], []
    for pid in part_list:
        # list will in-place edit
        pred_box = t.nodes[pid]['pred_box'].copy()
        gt_box = t.nodes[pid]['gt_box'].copy()
        if use_gt_rot:
            pred_box[6:] = gt_box[6:]
        if use_gt_size:
            pred_box[3:6] = match_gt_size(pred_box, gt_box)
        if use_gt_center:
            pred_box[:3] = gt_box[:3]
        # print(f'prediction {pid}, {pred_box}')

        pred_box_list.append(pred_box)
        box_color_list.append(t.nodes[pid]['color'])

    # pred_ply_fn = Path(output_dir)/'pred.ply'
    logger.debug(f'Output pred ply to {pred_ply_fn}')
    convert_box_to_ply(pred_box_list,
                       box_color_list,
                       str(pred_ply_fn))


def logger_obj_html_unary(curr_obj, out_file, loss_list):
    """Build visulizer tool with prediction and gt visualizing materials"""

    img_list = [['./img.jpg', './mask_img.jpg']]
    ply_list = ['./pred.ply']
    loss_list = [0.0]

    img_list.append(['./img.jpg', './mask_img.jpg'])
    ply_list.append('./gt.ply')
    loss_list.append(0.0)

    ply_name_list = [f"\'{Path(item).stem}\'" for item in ply_list]  # 'ply_name' not ply_name
    logger.info(f'Save html to {out_file}')
    get_ncanvas_html(img_list,
                     ply_list,
                     ply_name_list,
                     curr_obj,
                     'TouchIDNet',
                     ncanvas_template_file,
                     out_file,
                     loss_list=loss_list)


def build_update_part_info(cfg):
    if cfg.MODEL.CHOICE in unary_rot_model:
        def update_part_info(preds, data_batch, G, cfg):
            pid_list = data_batch['pids'][0].cpu().numpy()
            preds_xydir = preds['xydir'].detach().cpu().numpy()
            for i, pid in enumerate(pid_list):
                G.nodes[pid]['pred_box'][6:] = preds_xydir[i, :]
                logger.debug(f"Update part rot {pid}, {G.nodes[pid]['pred_box']}")
                # print(f'update rotaion {preds_xydir[i, :]}')
        return update_part_info
    elif cfg.MODEL.CHOICE in unary_size_model:
        def update_part_info(preds, data_batch, G, cfg):
            pid_list = data_batch['pids'][0].cpu().numpy()
            preds_size = preds['size'].detach().cpu().numpy()
            for i, pid in enumerate(pid_list):
                G.nodes[pid]['pred_box'][3:6] = preds_size[i, :]
                logger.debug(f"Update part size {pid}, {G.nodes[pid]['pred_box']}")
                # print(f'update size {preds_size[i, :]}')
        return update_part_info
    else:
        raise NotImplementedError


def get_preds_gt_option(cfg):
    if cfg.MODEL.CHOICE in unary_rot_model:
        gt_rot, gt_size, gt_center = False, True, True
        return gt_rot, gt_size, gt_center
    elif cfg.MODEL.CHOICE in unary_size_model:
        gt_rot, gt_size, gt_center = False, False, True
        return gt_rot, gt_size, gt_center
    else:
        raise NotImplementedError()


def process_obj_parts(obj_fn,
                      model,
                      loss_fn,
                      metric_fn,
                      output_dir,
                      cfg,
                      args_choice='max'):
    # pred use gt choice
    update_part_info = build_update_part_info(cfg)
    gt_rot, gt_size, gt_center = get_preds_gt_option(cfg)

    # read shape graph
    t = nx.read_gpickle(obj_fn)
    assert type(t) == ShapeGraph

    curr_obj = Path(obj_fn).stem
    logger.info(f'Process {curr_obj}')

    loss_key, metric_key = get_loss_metric_key(cfg)

    data_batch = get_parts_batch(t)
    data_batch = {
        k: v.cuda(non_blocking=True)
        for k, v in data_batch.items()
        if isinstance(v, torch.Tensor)
    }
    preds, losses, metrics = test_model(data_batch,
                                        model,
                                        loss_fn,
                                        metric_fn,
                                        loss_key=loss_key,
                                        metric_key=metric_key)

    curr_loss = float(losses[0])
    curr_metric = float(metrics[0])

    logger.debug(f'Log graph1')
    t.log_graph()

    pid_list = data_batch['pids'][0].cpu().numpy()
    update_part_info(preds, data_batch, t, cfg)

    # visualizer
    pred_ply_fn = Path(output_dir) / 'pred.ply'
    logger_pred_unary(t,
                      pid_list,
                      pred_ply_fn,
                      use_gt_rot=gt_rot,
                      use_gt_size=gt_size,
                      use_gt_center=gt_center)

    logger_gt(t, pid_list, output_dir)

    out_file = Path(output_dir, f'{curr_obj}.html')
    logger_obj_html_unary(curr_obj,
                          out_file=out_file,
                          loss_list=[0.0])

    # write updated shape graph result
    logger.info(f'Write updated graph to {obj_fn}')
    nx.write_gpickle(t, obj_fn)

    obj_npy_fn = output_dir/f'{curr_obj}.npy'
    logger.info(f'Write predicted box dict to {obj_npy_fn}')
    t.output_npy(out_pred_fn=obj_npy_fn)

    logger.debug(f'Updated graph')
    t.log_graph()

    return curr_loss, curr_metric
