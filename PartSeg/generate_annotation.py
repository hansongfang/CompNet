"""Generate annotations for detectron2.

Reference: https://detectron2.readthedocs.io/tutorials/datasets.html#custom-dataset-dicts-for-new-tasks
"""
import os
import glob
import argparse
import cv2
import random
import numpy as np
import pycocotools.mask
import tqdm
import pickle
from detectron2.structures.boxes import BoxMode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir", type=str, help="root directory", default="datasets/chair/images"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="path to output",
        default="datasets/chair/annotations/instances.pkl",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--shape-list-fn", type=str, help="path to shape list")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_dir = args.root_dir

    height, width = args.height, args.width

    if args.shape_list_fn:
        with open(args.shape_list_fn, "r") as f:
            image_ids = f.readlines()
        image_ids = [x.strip() for x in image_ids]
    else:
        image_ids = os.listdir(root_dir)

    image_ids = sorted(image_ids)
    if args.random:
        random.seed(0)
        random.shuffle(image_ids)
    image_ids = image_ids[args.start : args.end]
    print(f"Processing {len(image_ids)} images.")

    dataset = []
    for image_id in tqdm.tqdm(image_ids[:]):
        # print(image_id)
        file_name = os.path.join(root_dir, image_id, "img.png")
        if args.vis:
            image = cv2.imread(file_name)

        annotations = []
        mask_file_names = glob.glob(os.path.join(root_dir, image_id, "partmask*.png"))
        for mask_file_name in mask_file_names:
            basename = os.path.splitext(os.path.basename(mask_file_name))[0]
            part_id = int(basename.split("_")[-1])

            # axis aligned box
            mask = cv2.imread(mask_file_name)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            # Pass empty mask
            if not (mask > 0).any():
                # print(mask_file_name)
                continue

            bbox = cv2.boundingRect(mask)
            x, y, w, h = bbox
            mask_encoded = pycocotools.mask.encode(np.asarray(mask > 0, order="F"))
            assert w > 0 and h > 0, mask_file_name

            if args.vis:
                image_to_show = image.copy()
                x, y, w, h = bbox
                cv2.rectangle(image_to_show, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.imshow("image", image_to_show)
                if cv2.waitKey(0) == 27:  # Esc key to stop
                    break
                cv2.imshow("mask", mask)
                if cv2.waitKey(0) == 27:  # Esc key to stop
                    break

            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 0,
                "segmentation": mask_encoded,
                "id": part_id,
            }
            annotations.append(obj)

        data_dict = {
            "file_name": os.path.join(
                image_id, "img.png"
            ),  # require to provide root_dir when loading
            "height": height,
            "width": width,
            "image_id": image_id,
            "annotations": annotations,
        }
        dataset.append(data_dict)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
