from pathlib import Path
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader

from CompNet.utils.io_utils import load_pickle, read_img, resize_img, get_list_from_file


class ImagePartDataset(Dataset):
    def __init__(self,
                 renderer_dir,
                 part_list_file="",
                 shape_list_file="",
                 img_height=256,
                 img_width=256,
                 mask_prefix='partmask',
                 dataset_name='train'):
        self.dataset_name = dataset_name
        self.data_dir = Path(renderer_dir)
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.mask_prefix = mask_prefix

        if self.dataset_name == "test":
            self.shape_list = get_list_from_file(str(Path(renderer_dir)/shape_list_file))
            self.shape_list = self.shape_list[:50]
            print(f'{dataset_name}: Image size {self.IMG_HEIGHT} {self.IMG_WIDTH}, {len(self.shape_list)} shapes')
        else:
            self.part_list = load_pickle(str(Path(renderer_dir)/part_list_file))
            random.shuffle(self.part_list)
            print(f'{dataset_name}: Image size {self.IMG_HEIGHT} {self.IMG_WIDTH}, {len(self.part_list)} parts')

    def get_item_test(self, index):
        assert self.dataset_name == 'test'
        obj = self.shape_list[index]

        obj_file = self.data_dir / obj / f'{obj}.npy'

        img_file = self.data_dir / obj / 'img.png'
        assert img_file.exists()

        # img
        img = read_img(str(img_file))
        height, width, _ = img.shape
        if height != self.IMG_HEIGHT:
            img = resize_img(img, self.IMG_HEIGHT, self.IMG_WIDTH)

        # partmask partbox
        obj_data = np.load(obj_file, allow_pickle=True).item()
        pid_list = obj_data['vis_parts']
        parts_mask = []
        parts_box = []
        parts_id = []
        for pid in pid_list:
            mask_file = self.data_dir / obj / f'{self.mask_prefix}_{pid}.png'
            mask = read_img(str(mask_file), 0)
            height, width = mask.shape
            if height != self.IMG_HEIGHT:
                mask = resize_img(mask, self.IMG_HEIGHT, self.IMG_WIDTH)
            parts_mask.append(mask)

            part_box = obj_data['nodes'][pid]['box']
            parts_box.append(part_box)

            parts_id.append(pid)

        parts_mask = np.array(parts_mask)
        parts_box = np.array(parts_box)
        parts_id = np.array(parts_id)

        img = torch.tensor(img.copy()).permute(2, 0, 1).float()  # (3, height , width)
        parts_mask = torch.tensor(parts_mask).float()  # (num_part, height, width)
        boxes = torch.tensor(parts_box).float()
        gt_box = torch.tensor(parts_box).float()  # (num_part, 12)
        parts_id = torch.tensor(parts_id).int()  # (num_parts, )
        weight = torch.ones_like(parts_id).float()  # num_parts

        return {
            'img': img,
            'obj_name': obj,
            'pids': parts_id,
            'mask': parts_mask,
            'boxes': boxes,
            'gt_box': gt_box,
            'weight': weight,
        }

    def __getitem__(self, index):
        if self.dataset_name == 'test':
            return self.get_item_test(index)  # test mode return parts in one image

        obj, id1 = self.part_list[index]

        obj_file = self.data_dir / obj / f'{obj}.npy'
        img_file = self.data_dir / obj / 'img.png'
        mask_file = self.data_dir / obj / f'{self.mask_prefix}_{id1}.png'

        img = read_img(str(img_file))
        height, width, _ = img.shape
        if height != self.IMG_HEIGHT:
            img = resize_img(img, self.IMG_HEIGHT, self.IMG_WIDTH)

        mask = read_img(str(mask_file), 0)
        height, width = mask.shape
        if height != self.IMG_HEIGHT:
            mask = resize_img(mask, self.IMG_HEIGHT, self.IMG_WIDTH)

        obj_data = np.load(obj_file, allow_pickle=True).item()
        box = obj_data['nodes'][id1]['box']

        weight = 1.0

        img = torch.tensor(img.copy()).permute(2, 0, 1).float()  # 3 * height * width
        part_mask = torch.tensor(mask).float().unsqueeze(dim=0)  # one_part * height * width
        part_box = torch.tensor(box).float().unsqueeze(dim=0)  # one_part * 12
        part_weight = torch.tensor(weight).float().unsqueeze(dim=0)  # one_part,
        gt_box = torch.tensor(box).float().unsqueeze(dim=0)  # one_part * 12

        return {
            'img': img,
            'obj_name': f'{obj}-part{id1}',
            'mask': part_mask,
            'boxes': part_box,
            'gt_box': gt_box,
            'weight': part_weight,
        }

    def __len__(self):
        if self.dataset_name == "test":
            return len(self.shape_list)
        else:
            return len(self.part_list)


def build_data_loader(cfg, mode="train"):
    if mode == "train":
        dataset = ImagePartDataset(
            renderer_dir=cfg.DATA.PART.TRAIN.IMG_DIR,
            part_list_file=cfg.DATA.PART.TRAIN.PART_LIST,
            img_height=cfg.DATA.PART.TRAIN.IMG_HEIGHT,
            img_width=cfg.DATA.PART.TRAIN.IMG_WIDTH,
            mask_prefix=cfg.DATA.PART.TRAIN.MASK_PREFIX,
            dataset_name=mode,
        )
    elif mode == "val":
        dataset = ImagePartDataset(
            renderer_dir=cfg.DATA.PART.VAL.IMG_DIR,
            part_list_file=cfg.DATA.PART.VAL.PART_LIST,
            img_height=cfg.DATA.PART.VAL.IMG_HEIGHT,
            img_width=cfg.DATA.PART.VAL.IMG_WIDTH,
            mask_prefix=cfg.DATA.PART.VAL.MASK_PREFIX,
            dataset_name=mode,
        )
    elif mode == "test":
        dataset = ImagePartDataset(
            renderer_dir=cfg.DATA.PART.TEST.IMG_DIR,
            shape_list_file=cfg.DATA.PART.TEST.SHAPE_LIST,
            img_height=cfg.DATA.PART.TEST.IMG_HEIGHT,
            img_width=cfg.DATA.PART.TEST.IMG_WIDTH,
            mask_prefix=cfg.DATA.PART.TEST.MASK_PREFIX,
            dataset_name=mode,
        )
    else:
        raise ValueError("Unknown mode: {}.".format(mode))

    if mode == "train" or mode == "val":
        batch_size = cfg.TRAIN.BATCH_SIZE
    else:
        batch_size = cfg.TEST.BATCH_SIZE

    data_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.DATA.NUM_WORKERS,
    )

    return data_loader
