from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import networkx as nx
from loguru import logger
import argparse
import numpy as np
np.set_printoptions(precision=2, suppress=True)

import torch
import torch.nn as nn

from CompNet.utils.io_utils import get_list_from_file
from CompNet.models.build_model import build_model
from common.tools.checkpoint import Checkpointer
from CompNet.config import load_cfg_from_file
from CompNet.file_logger import logger_table_html2

from CompNet.test_misc.shape_graph import EqualSizeGraph, ShapeGraph
from CompNet.test_misc.process_tp import get_tp_batch_v3 as get_tp_batch
from CompNet.test_misc.process_part import logger_pred_unary, logger_obj_html_unary
from CompNet.test_misc.common import logger_gt, test_model
from CompNet.test_misc.shape_graph import init_shape_graph
table_template_path = './CompNet/viewer/templates/table.html'


def get_parts_batch_v2(t, pid_list=[]):
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
    parts_img = []
    for pid in pid_list:
        mask = t.nodes[pid]['mask_img']
        box = t.nodes[pid]['pred_box']
        logger.debug(f'Loading part {pid} box, {box}')
        gt_box = t.nodes[pid]['gt_box']
        parts_mask.append(mask)
        parts_box.append(box)
        gt_boxes.append(gt_box)
        parts_id.append(pid)
        parts_img.append(img)

    parts_mask = np.array(parts_mask)
    parts_box = np.array(parts_box)
    gt_boxes = np.array(gt_boxes)
    parts_id = np.array(parts_id)
    parts_img = np.array(parts_img)

    parts_img = torch.tensor(parts_img.copy())
    parts_img = parts_img.permute(0, 3, 1, 2).float().unsqueeze(0)
    parts_mask = torch.tensor(parts_mask).float().unsqueeze(0)  # batch_size, num_part, height, width
    parts_box = torch.tensor(parts_box).float().unsqueeze(0)  # batch_size, num_part, 12
    gt_boxes = torch.tensor(gt_boxes).float().unsqueeze(0)  # batch_size, num_part, 12
    parts_id = torch.tensor(parts_id).int().unsqueeze(0)  # batch_size, num_parts

    return {
        'img': parts_img,
        'obj_name': t.name,
        'pids': parts_id,
        'mask': parts_mask,
        'box': parts_box,
        'gt_box': gt_boxes
    }


def process_obj_group(obj_file,
                      model,
                      loss_fn,
                      metric_fn,
                      output_dir,
                      cfg,
                      args_choice='max',
                      threshold=0.9):
    model.eval()
    metric_fn.eval()
    loss_fn.eval()

    # read graph
    t = nx.read_gpickle(obj_file)
    assert type(t) == ShapeGraph

    step_choices = t.get_all_pairs()
    if len(step_choices) == 0:
        raise ValueError(f'Empty pairs')

    data_batch = get_tp_batch(t, step_choices)
    batch_size = data_batch['img'].shape[0]

    data_batch = {
        k: v.cuda(non_blocking=True)
        for k, v in data_batch.items()
        if isinstance(v, torch.Tensor)
    }

    loss_key = 'loss_bce'
    metric_key = 'me-acc'
    preds, losses, metrics = test_model(data_batch,
                                        model,
                                        loss_fn,
                                        metric_fn,
                                        loss_key=loss_key,
                                        metric_key=metric_key)

    preds_equalsize = preds['adj'].squeeze(1).detach().cpu().numpy()
    for i in range(batch_size):
        id1, id2 = data_batch['ids'][i].cpu().numpy()
        logger.debug(f'pair {id1}-{id2}, preds: {preds_equalsize[i]:.4f}')

    curr_loss = float(losses[0])
    curr_metric = float(metrics[0])

    size_graph = EqualSizeGraph()
    node_list = list(t.nodes)
    size_graph.add_nodes_from(node_list)
    logger.info(f'Judge equal size threshold {threshold}')
    for i in range(batch_size):
        if preds_equalsize[i] > threshold:
            # equal size
            id1, id2 = data_batch['ids'][i].cpu().numpy()
            id1, id2 = int(id1), int(id2)
            size_graph.add_edge(id1, id2)
            logger.debug(f'Add equal size edge pair {id1}-{id2}')

    num_comp, comp_list = size_graph.get_number_of_components()
    size_graph.log_graph()

    t.node_size_group = comp_list
    nx.write_gpickle(t, obj_file)
    for item in comp_list:
        logger.info(f'component: {item}')
    return curr_loss, curr_metric


def update_part_info(preds, data_batch, G, cfg):
    pid_list = data_batch['pids'][0].cpu().numpy()
    preds_size = preds['size'][0].detach().cpu().numpy()
    box1 = data_batch['box'][0, 0, :].cpu().numpy()

    # parallel equal
    # both rotation and size should be assigned
    for i, pid in enumerate(pid_list):
        G.nodes[pid]['pred_box'][3:6] = preds_size[i, :]
        G.nodes[pid]['pred_box'][6:] = box1[6:]
        logger.debug(f"Update part size {pid}, {G.nodes[pid]['pred_box']}")
    return update_part_info


def process_obj_nparts_size(obj_file,
                            model,
                            loss_fn,
                            metric_fn,
                            output_dir,
                            cfg,
                            args_choice='max'):
    # read shape graph
    logger.debug(f'Print 1')
    t = nx.read_gpickle(obj_file)
    assert type(t) == ShapeGraph

    curr_obj = Path(obj_file).stem
    print(f'Process {curr_obj}')

    loss_key = 'loss-l2'
    metric_key = 'me-cd'

    obj_size_group = t.node_size_group
    for one_group in obj_size_group:

        data_batch = get_parts_batch_v2(t, pid_list=one_group)
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

        logger.debug(f'Before updating size')
        t.log_graph()
        pid_list = data_batch['pids'][0].cpu().numpy()
        update_part_info(preds, data_batch, t, cfg)
        logger.debug(f'After updating size')
        t.log_graph()

    # visualizer
    pred_ply_fn = Path(output_dir) / 'pred.ply'
    total_pid_list = list(t.nodes)
    logger_pred_unary(t,
                      part_list=total_pid_list,
                      pred_ply_fn=pred_ply_fn,
                      use_gt_rot=False,
                      use_gt_size=False,
                      use_gt_center=True)

    logger_gt(t, total_pid_list, output_dir)

    out_file = Path(output_dir, f'{curr_obj}.html')
    logger_obj_html_unary(curr_obj,
                          out_file=out_file,
                          loss_list=[0.0])

    # write updated shape graph result
    logger.info(f'Write updated graph to {obj_file}')
    nx.write_gpickle(t, obj_file)

    obj_npy_fn = output_dir / f'{curr_obj}.npy'
    logger.info(f'Write predicted box dict to {obj_npy_fn}')
    t.output_npy(out_pred_fn=obj_npy_fn)

    logger.debug(f'Updated graph')
    t.log_graph()

    return curr_loss, curr_metric


def load_model_from_cfg(cfg, output_dir=""):
    # build_model
    model, loss_fn, metric_fn = build_model(cfg, mode='test')
    logger.info("Build model:\n{}".format(str(model)))
    model = nn.DataParallel(model).cuda()

    # build checkpoint
    checkpointer = Checkpointer(model,
                                save_dir=output_dir,
                                logger=logger)

    if cfg.TEST.WEIGHT:
        weight_path = cfg.TEST.WEIGHT.replace("@", str(output_dir))
        logger.info(f'Loading weight from {weight_path}')
        checkpointer.load(weight_path, resume=False)
    else:
        raise ValueError(f'None weight path!')

    return model, loss_fn, metric_fn


def build_process_obj_size(cfg):
    if cfg.MODEL.CHOICE == 'SizeRelationNet':
        return process_obj_group
    elif cfg.MODEL.CHOICE == 'GroupAxisLengthNet':
        return process_obj_nparts_size
    else:
        raise NotImplementedError()


def test(cfg, obj_test_list, shape_graph_dir, output_dir="", **kwargs):
    """Get essential model, loss_fn, metric"""

    model, loss_fn, metric_fn = load_model_from_cfg(cfg, output_dir)
    process_obj = build_process_obj_size(cfg)

    all_result = []
    for obj in obj_test_list:
        obj_file = shape_graph_dir / f'{obj}.gpickle'
        curr_obj = obj_file.stem

        try:
            obj_out_dir = Path(output_dir) / curr_obj
            obj_out_dir.mkdir(exist_ok=True)
            args_choice = 'max'
            logger.info(f'Process {obj} with group size.')
            curr_loss, curr_metric = process_obj(obj_file,
                                                 model,
                                                 loss_fn,
                                                 metric_fn,
                                                 obj_out_dir,
                                                 cfg=cfg,
                                                 args_choice=args_choice,
                                                 **kwargs)
            all_result.append([curr_obj, curr_loss, curr_metric])
        except:
            print(f'Not able to process shape {curr_obj}, empty nodes or contain one node. Due to visibility issue.')

    # logger table html
    out_file = Path(output_dir) / f'table.html'
    logger_table_html2(all_result, str(out_file), table_template_path)


def parse_args():
    parser = argparse.ArgumentParser(description="TouchParts test")
    parser.add_argument(
        "--cfg_sizerelation",
        default="./configs/SizeRelationNet.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--cfg_groupsize",
        default="./configs/GroupAxisLengthNet.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--shape_list",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--render_dir",
        metavar="FILE",
        help="path to render directory",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs/test_nparts_size",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        '--start_id',
        help='Test part of object list',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--end_id',
        help='Test part of object list',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--thresh_size',
        type=float,
        default=0.9)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def main_group_size(args, shape_list, shapeGraph_dir, output_dir):
    # perform size cluster
    cfg_file = args.cfg_sizerelation
    cfg = load_cfg_from_file(cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    logger_out_dir = Path(output_dir) / 'size'
    logger_out_dir.mkdir(exist_ok=True, parents=True)
    test(cfg, shape_list, shapeGraph_dir, str(logger_out_dir), threshold=args.thresh_size)
    cfg.defrost()

    # perform group size estimation
    cfg_file = args.cfg_groupsize
    cfg = load_cfg_from_file(cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    logger_out_dir = Path(output_dir) / 'size'
    logger_out_dir.mkdir(exist_ok=True, parents=True)
    test(cfg, shape_list, shapeGraph_dir, str(logger_out_dir))
    cfg.defrost()


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()
    start_id = args.start_id
    end_id = args.end_id

    output_dir = args.output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    # inital shape graph
    logger.info('Initial shape graph')
    shape_list_fn = args.shape_list
    shape_list = get_list_from_file(shape_list_fn)
    if end_id != 0:
        shape_list = shape_list[start_id:end_id]
    logger.info(f'Process {len(shape_list)} shapes, from start_id {start_id} to end_id {end_id}.')

    render_dir = args.render_dir
    shapeGraph_dir = output_dir / 'shapeGraph'
    shapeGraph_dir.mkdir(exist_ok=True, parents=True)
    init_shape_graph(render_dir, shape_list, shapeGraph_dir)

    # use gt rotation
    logger.info(f'Use gt rotation')
    for obj in shape_list:
        obj_file = Path(shapeGraph_dir) / f'{obj}.gpickle'
        t = nx.read_gpickle(obj_file)
        t.pred_use_gt(choice='rot')
        nx.write_gpickle(t, obj_file)

    # perform size cluster
    cfg_file = args.cfg_sizerelation
    cfg = load_cfg_from_file(cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    logger_out_dir = Path(output_dir)/'size'
    logger_out_dir.mkdir(exist_ok=True, parents=True)
    test(cfg, shape_list, shapeGraph_dir, logger_out_dir)
    cfg.defrost()

    # perform group size estimation
    cfg_file = args.cfg_groupsize
    cfg = load_cfg_from_file(cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    logger_out_dir = Path(output_dir) / 'size'
    logger_out_dir.mkdir(exist_ok=True, parents=True)
    test(cfg, shape_list, shapeGraph_dir, logger_out_dir)
    cfg.defrost()

    # perform center
    logger.info(f'Use gt center')
    for obj in shape_list:
        obj_file = Path(shapeGraph_dir) / f'{obj}.gpickle'
        t = nx.read_gpickle(obj_file)
        t.pred_use_gt(choice='center')
        nx.write_gpickle(t, obj_file)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()
