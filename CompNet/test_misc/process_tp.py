from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import os.path as osp

import numpy as np
np.set_printoptions(precision=2, suppress=True)
import networkx as nx
from loguru import logger

import torch
from torch.utils.data import Dataset, DataLoader

from CompNet.test_misc.shape_graph import ShapeGraph
from CompNet.utils.io_utils import write_img
from CompNet.test_misc.common import test_model
from CompNet.viewer.tool import get_ncanvas_html
from CompNet.test_misc.common import logger_gt, ncanvas_template_file
from CompNet.test_misc.common import get_loss_metric_key
from CompNet.test_misc.process_part import logger_pred_unary


def get_tp_batch(t, tp_list=[]):
    """
    Get touch pair data batch

    Args:
        t: shapeGraph,
        tp_list: touchpair list,

    Returns:

    """
    batch_imgs = []
    batch_boxes = []
    batch_gt_boxes = []
    batch_masks = []
    batch_ids = []
    batch_touch_label = []
    for (id1, id2) in tp_list:
        box1 = t.nodes[id1]['pred_box']
        box2 = t.nodes[id2]['pred_box']

        boxes = np.array([box1, box2])

        imgs = np.array([t.img, t.img])  # 2, height, width, 3
        masks = np.array([t.nodes[id1]['mask_img'],
                          t.nodes[id2]['mask_img']])  # 2, height, width
        logger.debug(f'Loading tp {id1}-{id2}, \n'
                     f'box1: {boxes[0]}\n'
                     f'box2: {boxes[1]}')
        gt_boxes = np.array([t.nodes[id1]['gt_box'],
                             t.nodes[id2]['gt_box']])  # 2, 12

        img = torch.tensor(imgs).permute(0, 3, 1, 2).float()  # 2, 3 * height * width
        parts_mask = torch.tensor(masks).float()  # 2 * height * width
        parts_box = torch.tensor(boxes).float()  # 2 * 12
        gt_box = torch.tensor(gt_boxes).float()
        edge_pair = torch.tensor([id1, id2]).int()
        label = torch.tensor([1.0]).float()

        batch_imgs.append(img)
        batch_boxes.append(parts_box)
        batch_gt_boxes.append(gt_box)
        batch_masks.append(parts_mask)
        batch_ids.append(edge_pair)
        batch_touch_label.append(label)

    batch_imgs = torch.stack(batch_imgs).float()
    batch_boxes = torch.stack(batch_boxes).float()
    batch_gt_boxes = torch.stack(batch_gt_boxes).float()
    batch_masks = torch.stack(batch_masks).float()
    batch_ids = torch.stack(batch_ids).int()
    batch_touch_label = torch.stack(batch_touch_label).float()

    return {'img': batch_imgs,
            'mask': batch_masks,
            'box': batch_boxes,
            'gt_box': batch_gt_boxes,
            'ids': batch_ids,
            'adj': batch_touch_label,
            }


# time efficient get_data_batch
class TPBatchDataset(Dataset):

    def __init__(self, t, tp_list):
        self.t = t
        self.tp_list = tp_list

    def __getitem__(self, index):
        t = self.t
        tp_list = self.tp_list
        id1, id2 = tp_list[index]
        box1 = t.nodes[id1]['pred_box']
        box2 = t.nodes[id2]['pred_box']
        # order = 6
        # box1 = change_box_order(box1, order=order)
        # box2 = change_box_order(box2, order=order)
        boxes = np.array([box1, box2])
        imgs = np.array([t.img, t.img])  # 2, height, width, 3
        masks = np.array([t.nodes[id1]['mask_img'],
                          t.nodes[id2]['mask_img']])  # 2, height, width
        # boxes = np.array([t.nodes[id1]['pred_box'],
        #                   t.nodes[id2]['pred_box']])  # 2, 12
        logger.debug(f'Loading tp {id1}-{id2}, \n'
                     f'box1: {boxes[0]}\n'
                     f'box2: {boxes[1]}')
        gt_boxes = np.array([t.nodes[id1]['gt_box'],
                             t.nodes[id2]['gt_box']])  # 2, 12
        img = torch.tensor(imgs).permute(0, 3, 1, 2).float()  # 2, 3 * height * width
        parts_mask = torch.tensor(masks).float()  # 2 * height * width
        parts_box = torch.tensor(boxes).float()  # 2 * 12
        gt_box = torch.tensor(gt_boxes).float()
        edge_pair = torch.tensor([id1, id2]).int()
        label = torch.tensor([1.0]).float()
        return {'img': img,
                'mask': parts_mask,
                'box': parts_box,
                'gt_box': gt_box,
                'ids': edge_pair,
                'adj': label,
                }

    def __len__(self):
        return len(self.tp_list)


def get_tp_batch_v3(t, tp_list=[]):
    tp_dataset = TPBatchDataset(t, tp_list)
    dataloader = DataLoader(tp_dataset, batch_size=len(tp_list), num_workers=4)
    dataiter = iter(dataloader)
    data_batch = next(dataiter)
    return data_batch


def mask_color_img(img, mask, mask_color, alpha=0.6):
    """
    Args:
        img: np.array, (H, W, C)
        mask: np.array, (H, W)
        mask_color: np.array, (3,)
        alpha: float
    """
    mask_img = mask[:, :, np.newaxis] * mask_color
    return alpha * img + (1 - alpha) * mask_img


def logger_preds_tp(t,
                    part_list,
                    data_batch,
                    batch_id,
                    output_dir,
                    step,
                    use_gt_rot=False,
                    use_gt_size=False,
                    use_gt_center=False):
    """ Visualize each step. Input is image and two part mask, output part2"""
    # prediction 3D boxes
    pred_ply_file = str(Path(output_dir)/f'pred_step{step}.ply')
    logger_pred_unary(t,
                      part_list,
                      pred_ply_file,
                      use_gt_rot=use_gt_rot,
                      use_gt_size=use_gt_size,
                      use_gt_center=use_gt_center)

    # input image and masks
    masks = data_batch['mask']
    mask1 = masks[batch_id, 0, :, :].detach().cpu().numpy()  # part1
    mask2 = masks[batch_id, 1, :, :].detach().cpu().numpy()  # part2

    img = data_batch['img']
    img = img[0, 0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()

    id1, id2 = data_batch['ids'][batch_id].detach().cpu().numpy()
    mask1_img = mask_color_img(img, mask1, t.nodes[id1]['color'])
    mask2_img = mask_color_img(img, mask2, t.nodes[id2]['color'])

    write_img(str(Path(output_dir)/f'mask1_step{step}.jpg'), mask1_img)
    write_img(str(Path(output_dir)/f'mask2_step{step}.jpg'), mask2_img)


def logger_obj_html_tp(num_step, curr_obj, out_file, loss_list):
    """Build visulizer tool with prediction and gt visualizing materials"""

    img_list = [[f'mask1_step{i}.jpg', f'mask2_step{i}.jpg'] for i in range(1, num_step+1)]
    ply_list = [f'pred_step{i}.ply' for i in range(1, num_step+1)]

    img_list.append(['./img.jpg', './mask_img.jpg'])
    ply_list.append('./gt.ply')
    loss_list.append(0.0)

    ply_name_list = [f"\'{Path(item).stem}\'" for item in ply_list]  # 'ply_name' not ply_name
    print(f'Save to {out_file}')
    get_ncanvas_html(img_list,
                     ply_list,
                     ply_name_list,
                     curr_obj,
                     'TouchIDNet',
                     ncanvas_template_file,
                     out_file,
                     loss_list=loss_list)


def get_preds_gt_option(cfg):
    if cfg.MODEL.CHOICE == 'JointNet':
        gt_rot, gt_size, gt_center = False, False, False
        return gt_rot, gt_size, gt_center
    else:
        raise NotImplementedError()


def build_update_part_info(cfg):
    if cfg.MODEL.CHOICE == 'JointNet':
        def update_part_info(preds, batch_id, tp_id, G, cfg):
            u, v = tp_id

            relative_center = preds['offset'][batch_id, :]  # box2 offset in box1 coord
            relative_center = relative_center.detach().cpu().numpy()
            v_center = relative_center + G.nodes[u]['pred_box'][:3]
            G.nodes[v]['pred_box'][:3] = v_center

        return update_part_info
    else:
        raise NotImplementedError()


def process_obj_tp(obj_fn,
                   model,
                   loss_fn,
                   metric_fn,
                   output_dir,
                   cfg,
                   args_choice='max'):
    t = nx.read_gpickle(obj_fn)
    assert type(t) == ShapeGraph

    loss_key, metric_key = get_loss_metric_key(cfg)

    update_part_info = build_update_part_info(cfg)

    gt_rot, gt_size, gt_center = get_preds_gt_option(cfg)

    if cfg.MODEL.CHOICE == 'JointNet':
        s = t.init_start_node(gt_option='center')
    else:
        s = t.init_start_node(gt_option='none')
    tp_list = []
    loss_list = []
    metric_list = []
    part_list = [s]
    count_step = 0
    while not t.visit_all_nodes():
        logger.debug(f'Step choice: {args_choice}')
        step_choices = t.get_step_choices(option=args_choice)  # max mask area
        data_batch = get_tp_batch(t, step_choices)
        batch_size = data_batch['img'].shape[0]

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

        # for i in range(batch_size):
        #     id1, id2 = data_batch['ids'][i].cpu().numpy()
        #     logger.debug(f'pair {id1}-{id2} {metric_key}: {metrics[i]:.4f}')
        #
        # batch_id = np.argmin(metrics)
        batch_id = 0
        u, v = step_choices[batch_id]

        logger.debug(f'Visit node {v}, '
                    f'loss: {losses[batch_id]:.4f}, '
                    f'metric: {metrics[batch_id]:.4f}')
        tp_list.append(step_choices[batch_id])
        t.sf_update_graph(v)
        count_step += 1

        # update part pred box info
        update_part_info(preds,
                         batch_id,
                         tp_id=step_choices[batch_id],
                         G=t,
                         cfg=cfg)
        logger.debug(f"Update part {v}, {t.nodes[v]['pred_box']}")

        loss_list.append(losses[batch_id])
        metric_list.append(metrics[batch_id])
        part_list.append(v)

        # prediction visualization
        logger_preds_tp(t,
                        part_list,
                        data_batch,
                        batch_id,
                        output_dir,
                        count_step,
                        use_gt_rot=gt_rot,
                        use_gt_size=gt_size,
                        use_gt_center=gt_center)

    # sequential visualizer
    logger_gt(t,
              part_list,
              output_dir)
    out_file = osp.join(output_dir, f'{t.name}_seq.html')
    logger_obj_html_tp(count_step, t.name, out_file, loss_list)
    
    if count_step != 0:
        avg_loss = sum(loss_list) / count_step
        avg_metric = sum(metric_list) / count_step
    else:
        avg_loss = avg_metric = 0.0
    logger.info(f'Generate shape {t.name} with tp_list {tp_list}, avg loss: {avg_loss:.4f}')

    # write updated shape graph result
    logger.info(f'Write updated graph to {obj_fn}')
    nx.write_gpickle(t, obj_fn)

    obj_npy_fn = output_dir / f'{t.name}.npy'
    logger.info(f'Write predicted box dict to {obj_npy_fn}')
    t.output_npy(out_pred_fn=obj_npy_fn)

    return avg_loss, avg_metric
