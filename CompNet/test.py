import argparse
import os
import os.path as osp
import logging
import time
import sys

sys.path.insert(0, osp.dirname(__file__) + "/..")

import torch
import torch.nn as nn

from utils.io_utils import dump_pickle

from CompNet.config import load_cfg_from_file
from common.tools.logger import setup_logger
from common.tools.metric_logger import MetricLogger
from common.tools.tensorboard_logger import TensorboardLogger
from common.tools.checkpoint import Checkpointer

from CompNet.models.build_model import build_model
from CompNet.datasets.build_dataset import build_data_loader
from CompNet.extra_build import get_test_output_dirname, build_file_logger
from CompNet.file_logger import logger_table_html2, logger_table_relation_html


table_template_path = './CompNet/viewer/templates/table.html'


def parse_args():
    parser = argparse.ArgumentParser(description="partRecon Testing")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def test_model(model,
               loss_fn,
               metric_fn,
               data_loader,
               tensorboard_logger,
               file_logger,
               log_period=1,
               output_dir="",
               model_choice="",
               cfg=None):
    logger = logging.getLogger("partRecon.test")
    logger.info(f'Output to {output_dir}')
    meters = MetricLogger(delimiter="  ", )
    model.eval()
    metric_fn.eval()

    mean_losses = 0
    all_losses = {}
    with torch.no_grad():
        end = time.time()
        all_result = []
        for iteration, data_batch in enumerate(data_loader):
            data_time = time.time() - end

            batch_obj_name = data_batch["obj_name"]
            curr_obj = data_batch["obj_name"][0]
            data_batch = {
                k: v.cuda(non_blocking=True)
                for k, v in data_batch.items()
                if isinstance(v, torch.Tensor)
            }
            preds = model(data_batch)
            batch_loss_dict = loss_fn(preds, data_batch)
            loss_dict = {k: torch.mean(v) for (k, v) in batch_loss_dict.items()}
            batch_metric_dict = metric_fn(preds, data_batch)
            metric_dict = {k: torch.mean(v) for (k, v) in batch_metric_dict.items()}

            losses = sum(loss_dict.values())
            meters.update(loss=losses, **loss_dict, **metric_dict)
            mean_losses += losses

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            if iteration % log_period == 0:
                logger.info(
                    meters.delimiter.join(
                        ["iter: {iter:4d}",
                         "{meters}",
                         ]
                    ).format(
                        iter=iteration,
                        meters=str(meters),
                    )
                )

                tensorboard_logger.add_scalars(
                    loss_dict, iteration, prefix="test"
                )

                file_logger(
                    data_batch,
                    preds,
                    output_dir,
                    prefix="test",
                    obj_name=curr_obj,
                    cfg=cfg,
                )

                # Track first element loss and metric
                curr_losses_dict = {k: v[0] for (k, v) in batch_loss_dict.items()}
                curr_losses = float(sum(curr_losses_dict.values()))

                curr_metric_dict = {k: v[0] for (k, v) in batch_metric_dict.items()}
                curr_metrics = float(sum(curr_metric_dict.values()))

                if cfg.MODEL.CHOICE in ['ADJNet', 'SizeRelationNet']:
                    all_result.append([curr_obj, f"{float(preds['adj'][0]):.3f}", f"{float(data_batch['adj'][0]):.3f}"])
                    print(f'ADJNet result: ', all_result[-1])
                else:
                    all_result.append(
                        [curr_obj, f'{curr_losses:.3f}', f'{curr_metrics:.3f}'])  # append metrics instead of loss
                logger.info(f'Append shape {curr_obj} with loss {curr_losses:.3f}, metric {curr_metrics:.3f} to table.')
                tensorboard_logger.flush()

    total_iteration = data_loader.__len__()
    mean_losses /= total_iteration
    logger.info(f'Avg loss: {mean_losses}')

    # save all result
    out_res_fn = osp.join(output_dir, 'result')
    print(f'Save test result to {out_res_fn}')
    dump_pickle(out_res_fn, all_result)

    out_file = osp.join(output_dir, 'table.html')
    if cfg.MODEL.CHOICE in ['ADJNet', 'SizeRelationNet']:
        logger_table_relation_html(all_result, out_file, table_template_path)
    else:
        logger_table_html2(all_result, out_file, table_template_path)

    return meters


def test(cfg, output_dir="", logger_out_dir=""):
    logger = logging.getLogger('partRecon.tester')

    # build_model
    logger.info(f'Build model')
    model, loss_fn, metric_fn = build_model(cfg, mode="test")
    model = nn.DataParallel(model).cuda()

    # build checkpoint
    checkpointer = Checkpointer(model, save_dir=output_dir)

    if cfg.TEST.WEIGHT:
        weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        checkpointer.load(None, resume=True)

    logger.info(f'Build dataloader')
    # build data loader
    test_data_loader = build_data_loader(cfg, mode="test")

    # build tensorboard logger
    tensorboard_logger = TensorboardLogger(logger_out_dir)

    # build file logger
    file_logger = build_file_logger(cfg, mode="test")

    # test
    start_time = time.time()
    logger.info(f'Test model')
    test_model(model,
               loss_fn,
               metric_fn,
               test_data_loader,
               tensorboard_logger,
               file_logger=file_logger,
               log_period=cfg.TEST.LOG_PERIOD,
               output_dir=logger_out_dir,
               model_choice=cfg.MODEL.CHOICE,
               cfg=cfg)
    test_time = time.time() - start_time
    logger.info(f"Test forward time: {test_time:.2f}s")


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    cfg = load_cfg_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace("@", config_path)
        os.makedirs(output_dir, exist_ok=True)
    # File logger output dir
    logger_out_dir = osp.join(output_dir,
                              get_test_output_dirname(cfg, mode='test'))
    os.makedirs(logger_out_dir, exist_ok=True)

    logger = setup_logger("partRecon", output_dir, prefix="test")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg, output_dir, logger_out_dir=logger_out_dir)


if __name__ == "__main__":
    main()
