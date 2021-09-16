import argparse
import os
import os.path as osp
import logging
import time
import sys
sys.path.insert(0, osp.dirname(__file__) + '/..')

import torch
import torch.nn as nn

from common.tools.logger import setup_logger
from common.tools.torch_utils import set_random_seed
from common.tools.metric_logger import MetricLogger
from common.tools.tensorboard_logger import TensorboardLogger
from common.tools.solver import build_optimizer, build_scheduler
from common.tools.checkpoint import Checkpointer
from common.nn.freezer import print_frozen_modules_and_params, freeze_modules, freeze_params

from CompNet.config import load_cfg_from_file
from CompNet.models.build_model import build_model
from CompNet.datasets.build_dataset import build_data_loader


def parse_args():
    parser = argparse.ArgumentParser(description="PartReconstruction Training")
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


def train_model(model,
                loss_fn,
                metric_fn,
                data_loader,
                optimizer,
                curr_epoch,
                tensorboard_logger,
                log_period=1,
                output_dir="",
                cfg=None,
                ):
    logger = logging.getLogger("partRecon.train")
    meters = MetricLogger(delimiter="  ")
    model.train()

    # frozen modules
    frozen_modules = cfg.TRAIN.FROZEN_MODULES
    frozen_params = cfg.TRAIN.FROZEN_PARAMS
    freeze_modules(model, frozen_modules)
    freeze_params(model, frozen_params)
    print_frozen_modules_and_params(model, logger=logger)

    end = time.time()
    total_iteration = data_loader.__len__()

    for iteration, data_batch in enumerate(data_loader):
        data_time = time.time() - end
        data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}

        preds = model(data_batch)
        optimizer.zero_grad()

        loss_dict = loss_fn(preds, data_batch)
        metric_dict = metric_fn(preds, data_batch)
        losses = sum(loss_dict.values())
        meters.update(loss=losses, **loss_dict, **metric_dict)

        losses.backward()

        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        # log
        if iteration % log_period == 0:
            logger.info(
                meters.delimiter.join(
                    [
                        "EPOCH: {epoch:2d}",
                        "iter: {iter:4d}",
                        "{meters}",
                        "lr: {lr:.2e}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    epoch=curr_epoch,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

            tensorboard_logger.add_scalars(loss_dict,
                                           curr_epoch * total_iteration + iteration,
                                           prefix="train")
            tensorboard_logger.add_scalars(metric_dict,
                                           curr_epoch * total_iteration + iteration,
                                           prefix="train")
            tensorboard_logger.flush()

        tensorboard_logger.flush()

    return meters


def validate_model(model,
                   loss_fn,
                   metric_fn,
                   data_loader,
                   curr_epoch,
                   tensorboard_logger,
                   log_period=1,
                   output_dir="",
                   ):
    logger = logging.getLogger("partRecon.validate")
    meters = MetricLogger(delimiter="  ")
    model.eval()  # evaluation module
    metric_fn.eval()
    total_iteration = data_loader.__len__()

    end = time.time()
    with torch.no_grad():
        logger.info(f'Validate model with {len(data_loader)} datas.')
        for iteration, data_batch in enumerate(data_loader):
            data_time = time.time() - end
            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}

            preds = model(data_batch)
            loss_dict = loss_fn(preds, data_batch)
            metric_dict = metric_fn(preds, data_batch)

            losses = sum(loss_dict.values())
            meters.update(loss=losses, **loss_dict, **metric_dict)
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if iteration % log_period == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "EPOCH: {epoch:2d}",
                            "iter: {iter:4d}",
                            "{meters}",
                        ]
                    ).format(
                        epoch=curr_epoch,
                        iter=iteration,
                        meters=str(meters),
                    )
                )

                tensorboard_logger.add_scalars(loss_dict,
                                               curr_epoch * total_iteration + iteration,
                                               prefix="val")
                tensorboard_logger.add_scalars(metric_dict,
                                               curr_epoch * total_iteration + iteration,
                                               prefix="val")
                tensorboard_logger.flush()

    return meters


def train(cfg, output_dir="", gamma=None):
    logger = logging.getLogger("partRecon.trainer")

    # build model
    set_random_seed(cfg.RNG_SEED)
    model, loss_fn, metric_fn = build_model(cfg)
    logger.info("Build model:\n{}".format(str(model)))
    model = nn.DataParallel(model).cuda()

    # build optimizer
    optimizer = build_optimizer(cfg, model)

    # build lr scheduler
    scheduler = build_scheduler(cfg, optimizer)

    # build checkpointer
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                logger=logger)
    checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.AUTO_RESUME)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD
    if gamma:
        scheduler.gamma = gamma
        logger.info(f'Overwrite checkpointer gamma {scheduler.gamma}')
    logger.info(f'Scheduler gamma: {scheduler.gamma}')

    # build data loader
    train_data_loader = build_data_loader(cfg, mode="train")
    val_period = cfg.TRAIN.VAL_PERIOD
    val_data_loader = build_data_loader(cfg, mode="val") if val_period > 0 else None

    # build tensorboard logger (optionally by comment)
    tensorboard_logger = TensorboardLogger(output_dir)

    # train
    max_epoch = cfg.SCHEDULER.MAX_EPOCH
    start_epoch = checkpoint_data.get("epoch", 0)
    best_metric_name = "best_{}".format(cfg.TRAIN.VAL_METRIC)
    best_metric = checkpoint_data.get(best_metric_name, None)
    logger.info("Start training from epoch {}".format(start_epoch))
    for epoch in range(start_epoch, max_epoch):
        cur_epoch = epoch + 1
        start_time = time.time()
        train_meters = train_model(model,
                                   loss_fn,
                                   metric_fn,
                                   data_loader=train_data_loader,
                                   optimizer=optimizer,
                                   curr_epoch=epoch,
                                   tensorboard_logger=tensorboard_logger,
                                   log_period=cfg.TRAIN.LOG_PERIOD,
                                   output_dir=output_dir,
                                   cfg=cfg,
                                   )
        epoch_time = time.time() - start_time
        logger.info("Epoch[{}]-Train {}  total_time: {:.2f}s".format(
            cur_epoch, train_meters.summary_str, epoch_time))

        # checkpoint
        if cur_epoch % ckpt_period == 0 or cur_epoch == max_epoch:
            checkpoint_data["epoch"] = cur_epoch
            checkpoint_data[best_metric_name] = best_metric
            checkpointer.save("model_{:03d}".format(cur_epoch), **checkpoint_data)

        scheduler.step()
        # validate
        if val_period == 0:
            logger.info(f'Ignore validation during training.')
        else:
            if cur_epoch % val_period == 0 or cur_epoch == max_epoch:
                val_meters = validate_model(model,
                                            loss_fn,
                                            metric_fn,
                                            data_loader=val_data_loader,
                                            curr_epoch=epoch,
                                            tensorboard_logger=tensorboard_logger,
                                            log_period=cfg.TEST.LOG_PERIOD,
                                            output_dir=output_dir,
                                            )
                logger.info("Epoch[{}]-Val {}".format(cur_epoch, val_meters.summary_str))

                # best validation
                cur_metric = val_meters.meters[cfg.TRAIN.VAL_METRIC].global_avg
                if best_metric is None or cur_metric < best_metric:
                    best_metric = cur_metric
                    checkpoint_data["epoch"] = cur_epoch
                    checkpoint_data[best_metric_name] = best_metric
                    checkpointer.save("model_best", **checkpoint_data)

    logger.info("Best val-{} = {}".format(cfg.TRAIN.VAL_METRIC, best_metric))

    return model


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
        output_dir = output_dir.replace('@', config_path)
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("partRecon", output_dir, prefix="train")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, output_dir)


if __name__ == "__main__":
    main()
