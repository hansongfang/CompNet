# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from detectron2/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import shutil
import time
import cv2
import tqdm
import numpy as np

from detectron2.config import get_cfg
from partseg import add_partseg_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_partseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )

    # load weights from the default path
    if not args.custom_weights:
        default_weights = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        if os.path.exists(default_weights):
            print("Use the default weights.")
        cfg.MODEL.WEIGHTS = default_weights

    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/mask_rcnn_R_50_FPN_chair.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="root directory",
        default="datasets/images/test/chair",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="path to output",
        default="datasets/predictions/test/chair",
    )
    parser.add_argument("--shape-list-fn", type=str, help="path to shape list")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--custom-weights", action="store_true", help="whether to use custom weights"
    )
    parser.add_argument(
        "--include-image", action="store_true", help="whether to include input images"
    )
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--with-score", action="store_true")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)

    root_dir = args.root_dir
    if args.shape_list_fn:
        with open(args.shape_list_fn, "r") as f:
            image_ids = f.readlines()
        image_ids = [x.strip() for x in image_ids]
    else:
        image_ids = os.listdir(root_dir)
        image_ids = [x for x in image_ids if os.path.isdir(os.path.join(root_dir, x))]
    image_ids = sorted(image_ids)
    image_ids = image_ids[args.start : args.end]

    for image_id in tqdm.tqdm(image_ids[:]):
        file_name = os.path.join(root_dir, image_id, "img.png")
        image = read_image(file_name, format="BGR")
        predictions = predictor(image)
        instances = predictions["instances"].to("cpu")
        pred_masks = instances.pred_masks.numpy()  # [N, H, W]
        pred_masks = (pred_masks * 255).astype(np.uint8)
        # for pred_mask in pred_masks:
        #     cv2.imshow('mask', pred_mask)
        #     if cv2.waitKey(0) == 27:
        #         break  # esc to quit

        output_dir = os.path.join(args.output_dir, image_id)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # save
        for idx, pred_mask in enumerate(pred_masks):
            output_file_name = os.path.join(output_dir, f"partmask_{idx}.png")
            cv2.imwrite(output_file_name, pred_mask)

        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image_rgb = image[:, :, ::-1]
        visualizer = Visualizer(image_rgb, None, instance_mode=ColorMode.IMAGE)
        if not args.with_score:
            # workaround to suppress visualizing scores
            instances.remove("scores")
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        if args.vis:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit
        else:
            output_file_name = os.path.join(output_dir, f"partmask_all.png")
            vis_output.save(output_file_name)

        if args.include_image:
            shutil.copy(file_name, os.path.join(output_dir, "img.png"))
