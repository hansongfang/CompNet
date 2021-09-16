"""
Combination of predicting rotation, size, center together.
"""

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

from CompNet.utils.io_utils import get_list_from_file, save_list_to_file
from CompNet.models.build_model import build_model
from common.tools.checkpoint import Checkpointer
from CompNet.config import load_cfg_from_file
from CompNet.file_logger import logger_table_html2
from CompNet.test_misc.process_part import process_obj_parts
from CompNet.test_misc.process_tp import process_obj_tp
from CompNet.test_misc.common import unary_size_model, unary_rot_model
from CompNet.test_misc.shape_graph import init_shape_graph
from CompNet.test_nparts_size import main_group_size

table_template_path = './CompNet/viewer/templates/table.html'


def build_process_obj(cfg):
    if cfg.MODEL.CHOICE in unary_rot_model or cfg.MODEL.CHOICE in unary_size_model:
        return process_obj_parts
    elif cfg.MODEL.CHOICE in ['JointNet']:
        return process_obj_tp
    else:
        raise ValueError(f'Not supported model choice {cfg.MODEL.CHOICE}')


def get_html_suffix(cfg):
    if cfg.MODEL.CHOICE in unary_rot_model or cfg.MODEL.CHOICE in unary_size_model:
        return ''
    elif cfg.MODEL.CHOICE in ['JointNet']:
        return '_seq'
    else:
        raise ValueError(f'Not supported model choice {cfg.MODEL.CHOICE}')


def test(cfg, obj_test_list, shape_graph_dir, output_dir=""):
    """Get essential model, loss_fn, metric"""
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

    all_result = []
    # obj_test_list = ['46568_ry0']
    error_list = []
    for obj in obj_test_list:
        obj_fn = shape_graph_dir / f'{obj}.gpickle'
        curr_obj = obj_fn.stem

        # try:
        obj_out_dir = Path(output_dir) / curr_obj
        obj_out_dir.mkdir(exist_ok=True)
        process_obj = build_process_obj(cfg)
        args_choice = 'maxvolume'
        logger.info(f'Process {obj} with choice {args_choice}, used in tp prediction.')
        curr_loss, curr_metric = process_obj(obj_fn,
                                             model,
                                             loss_fn,
                                             metric_fn,
                                             obj_out_dir,
                                             cfg=cfg,
                                             args_choice=args_choice)
        all_result.append([curr_obj, curr_loss, curr_metric])
        # except:
        #     logger.error(f'Not able to process shape {curr_obj}, empty nodes or contain one node?')
        #     error_list.append(curr_obj)

    # save result
    out_res_fn = Path(output_dir) / 'result.npy'
    np.save(str(out_res_fn), all_result)

    # error list
    out_error_fn = Path(output_dir) / 'error_list.txt'
    logger.info(f'Fail to process {len(error_list)} shapes, save not processed shapes to {out_error_fn}')
    save_list_to_file(error_list, str(out_error_fn))

    # logger table html
    html_suffix = get_html_suffix(cfg)
    out_file = Path(output_dir) / f'table{html_suffix}.html'
    logger_table_html2(all_result, str(out_file), table_template_path, suffix=html_suffix)


def parse_args():
    parser = argparse.ArgumentParser(description="TouchParts test")
    parser.add_argument(
        "--cfg_rot",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--cfg_size",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--cfg_center",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--choice",
        default="rot size center",
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
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="",
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
        '--group_size',
        help='Prediction using group size',
        action='store_true',
    )
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
        '--thresh_size',
        type=float,
        default=0.9)
    parser.add_argument(
        '--thresh_rot',
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


def main_choice(choice, args, shape_list, shape_graph_dir, output_dir):
    def get_cfg(x):
        if x == 'rot':
            return args.cfg_rot
        elif x == 'size':
            return args.cfg_size
        elif x == 'center':
            return args.cfg_center
        else:
            raise ValueError(f'Not implement choice {x}')

    assert choice in ['rot', 'size', 'center']
    if choice in args.choice:
        logger.info(f'Predict {choice}')
        # predict shape rotation
        cfg_fn = get_cfg(choice)
        cfg = load_cfg_from_file(cfg_fn)
        cfg.merge_from_list(args.opts)
        cfg.freeze()

        logger_out_dir = Path(output_dir) / choice
        logger_out_dir.mkdir(exist_ok=True, parents=True)
        test(cfg, shape_list, shape_graph_dir, logger_out_dir)
        cfg.defrost()
    else:
        logger.info(f'Use gt {choice}')
        for obj in shape_list:
            obj_fn = Path(shape_graph_dir) / f'{obj}.gpickle'
            t = nx.read_gpickle(obj_fn)
            t.pred_use_gt(choice=choice)
            nx.write_gpickle(t, obj_fn)


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
    render_dir = args.render_dir
    shape_list_fn = args.shape_list
    if shape_list_fn == "":
        shape_list = [x.stem for x in Path(render_dir).iterdir() if x.is_dir()]
    else:
        shape_list = get_list_from_file(shape_list_fn)
    if end_id != 0:
        shape_list = shape_list[start_id:end_id]
    logger.info(f'Process {len(shape_list)} shapes, from start_id {start_id} to end_id {end_id}.')
    #     shape_list = shape_list[:1]
    shapeGraph_dir = output_dir / 'shapeGraph'
    shapeGraph_dir.mkdir(exist_ok=True, parents=True)
    init_shape_graph(render_dir, shape_list, shapeGraph_dir)

    # perform rotation
    main_choice('rot', args, shape_list, shapeGraph_dir, output_dir)

    # perform size
    group_size = args.group_size
    logger.info(f'Using group size {group_size}')
    if group_size:
        main_group_size(args, shape_list, shapeGraph_dir, output_dir)
    else:
        main_choice('size', args, shape_list, shapeGraph_dir, output_dir)

    # perform center
    main_choice('center', args, shape_list, shapeGraph_dir, output_dir)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()
