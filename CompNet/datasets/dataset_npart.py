"""
Loading various parts into dataset
"""
from pathlib import Path
import numpy as np
from loguru import logger

import torch
from torch.utils.data import Dataset, DataLoader

from CompNet.utils.io_utils import read_img, resize_img, load_pickle


class NPartDataset(Dataset):
    def __init__(self,
                 render_dir,
                 pair_list_file,
                 mask_prefix='partmask',
                 img_height=256,
                 img_width=256,
                 dataset_name='train'):
        self.render_dir = Path(render_dir)
        pair_list_file = self.render_dir/pair_list_file
        self.pair_list = load_pickle(str(pair_list_file))

        logger.info(f' Nparts dataloader, {len(self.pair_list)} pairs')

        self.mask_prefix = mask_prefix
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.dataset_name = dataset_name

        if self.dataset_name == 'test':
            self.pair_list = self.pair_list[:100]

    def __getitem__(self, index):
        part_list = self.pair_list[index]
        obj = part_list[0].split('.')[0]
        pid_list = part_list[1:]
        str_pid_list = [str(x) for x in pid_list]
        obj_name = f'{obj}-part{"-".join(str_pid_list)}'

        obj_file = self.render_dir/obj/f'{obj}.npy'
        obj_data = np.load(obj_file, allow_pickle=True).item()
        parts_mask = []
        parts_box = []
        parts_id = []
        for pid in pid_list:
            mask_file = self.render_dir / obj / f'{self.mask_prefix}_{pid}.png'
            mask = read_img(str(mask_file), 0)
            height, width = mask.shape
            if height != self.IMG_HEIGHT:
                mask = resize_img(mask, self.IMG_HEIGHT, self.IMG_WIDTH)
            parts_mask.append(mask)

            part_box = obj_data['nodes'][pid]['box']
            parts_box.append(part_box)

            parts_id.append(pid)

        img_file = self.render_dir / obj / 'img.png'
        assert img_file.exists()
        img = read_img(str(img_file))
        height, width, _ = img.shape
        if height != self.IMG_HEIGHT:
            img = resize_img(img, self.IMG_HEIGHT, self.IMG_WIDTH)

        parts_mask = np.array(parts_mask)
        parts_box = np.array(parts_box)
        parts_id = np.array(parts_id)

        num_part = len(pid_list)
        img = torch.tensor(img.copy()).permute(2, 0, 1).float()
        parts_img = img.unsqueeze(0).expand(num_part, 3, height, width)
        parts_img = parts_img.contiguous()
        parts_mask = torch.tensor(parts_mask).float()  # (num_part, height, width)
        boxes = torch.tensor(parts_box).float()
        gt_box = torch.tensor(parts_box).float()  # (num_part, 12)
        parts_id = torch.tensor(parts_id).int()  # (num_parts, )

        return {
            'img': parts_img,
            'obj_name': obj_name,
            'pids': parts_id,
            'mask': parts_mask,
            'box': boxes,
            'gt_box': gt_box,
        }

    def __len__(self):
        return len(self.pair_list)


def build_data_loader(cfg, mode='train'):
    if mode == 'train':
        dataset = NPartDataset(
            render_dir=cfg.DATA.NParts.TRAIN.RENDER_DIR,
            pair_list_file=cfg.DATA.NParts.TRAIN.PAIR_LIST,
            mask_prefix=cfg.DATA.NParts.TRAIN.MASK_PREFIX,
            img_height=cfg.DATA.NParts.TRAIN.IMG_HEIGHT,
            img_width=cfg.DATA.NParts.TRAIN.IMG_WIDTH,
            dataset_name="train",
        )
    elif mode == 'val':
        dataset = NPartDataset(
            render_dir=cfg.DATA.NParts.VAL.RENDER_DIR,
            pair_list_file=cfg.DATA.NParts.VAL.PAIR_LIST,
            mask_prefix=cfg.DATA.NParts.VAL.MASK_PREFIX,
            img_height=cfg.DATA.NParts.VAL.IMG_HEIGHT,
            img_width=cfg.DATA.NParts.VAL.IMG_WIDTH,
            dataset_name="train",
        )
    elif mode == 'test':
        dataset = NPartDataset(
            render_dir=cfg.DATA.NParts.TEST.RENDER_DIR,
            pair_list_file=cfg.DATA.NParts.TEST.PAIR_LIST,
            mask_prefix=cfg.DATA.NParts.TEST.MASK_PREFIX,
            img_height=cfg.DATA.NParts.TEST.IMG_HEIGHT,
            img_width=cfg.DATA.NParts.TEST.IMG_WIDTH,
            dataset_name="train",
        )
    else:
        raise NotImplementedError()

    if mode == "train":
        batch_size = cfg.TRAIN.BATCH_SIZE
    elif mode == "val":
        batch_size = cfg.VAL.BATCH_SIZE
    else:
        batch_size = cfg.TEST.BATCH_SIZE

    data_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.DATA.NUM_WORKERS,
    )

    return data_loader
