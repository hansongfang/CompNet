# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys
import time
import numpy as np
import torch
from os.path import join
import cv2


def setup_logger(name, save_dir, prefix="", timestamp=True,
                 use_std=True,
                 out_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    if use_std:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if save_dir:
        timestamp = time.strftime(".%m_%d_%H_%M_%S") if timestamp else ""
        prefix = "." + prefix if prefix else ""
        log_file = os.path.join(save_dir, "log{}.txt".format(prefix + timestamp))
        if out_file:
            log_file = out_file
        print(f'Save log to file {log_file}')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def shutdown_logger(logger):
    logger.handlers = []
