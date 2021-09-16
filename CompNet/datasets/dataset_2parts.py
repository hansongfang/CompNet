from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

import torch
from torch.utils.data import Dataset, DataLoader

from CompNet.utils.io_utils import read_img, resize_img, load_pickle


def read_mask_file(mask_file, pid, IMG_HEIGHT, IMG_WIDTH):
    if pid == -1:
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float)
    else:
        mask = read_img(mask_file, 0)
        height, width = mask.shape
        if height != IMG_HEIGHT:
            mask = resize_img(mask, IMG_HEIGHT, IMG_WIDTH)
    return mask


class PairPartDataset(Dataset):

    def __init__(self,
                 pair_list_file,
                 mask_prefix='partmask',
                 img_height=256,
                 img_width=256,
                 dataset_name='train'):
        self.render_dir = Path(pair_list_file).parent
        self.pair_list = load_pickle(str(pair_list_file))

        # common param
        self.mask_prefix = mask_prefix
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.dataset_name = dataset_name

        if self.dataset_name == 'test':
            self.pair_list = self.pair_list[:100]
        logger.info(f'Two parts dataloader, {len(self.pair_list)} pairs')
        print('sample pair list: ', self.pair_list[0])

    def __getitem__(self, index):
        obj, id1, id2 = self.pair_list[index]
        # print(obj, id1, id2)
        obj_name = Path(obj).stem
        shape_id = obj_name.split('_r')[0]

        img_dir = self.render_dir / obj_name

        # loading images
        img_file = img_dir / 'img.png'
        img = read_img(str(img_file))
        height, width, _ = img.shape
        if height != self.IMG_HEIGHT:
            img = resize_img(img, self.IMG_HEIGHT, self.IMG_WIDTH)

        mask1_file = str(img_dir/f'{self.mask_prefix}_{id1}.png')
        mask1 = read_mask_file(mask1_file, id1, self.IMG_HEIGHT, self.IMG_WIDTH)

        mask2_file = str(img_dir/f'{self.mask_prefix}_{id2}.png')
        mask2 = read_mask_file(mask2_file, id2, self.IMG_HEIGHT, self.IMG_WIDTH)

        imgs = np.array([img, img])  # 2, height, width, 3
        masks = np.array([mask1, mask2])  # 2, height, width

        # loading box
        obj_file = img_dir / f'{obj_name}.npy'
        assert obj_file.exists()
        obj_data = np.load(str(obj_file), allow_pickle=True).item()
        box1 = obj_data['nodes'][id1]['box']
        box2 = obj_data['nodes'][id2]['box']
        boxes = np.array([box1, box2])  # 2, 12

        img = torch.tensor(imgs).permute(0, 3, 1, 2).float()  # 2, 3 * height * width
        parts_mask = torch.tensor(masks).float()  # 2 * height * width
        # ! we use gt box for training, at testing ,we use prediction box from previous stage
        # num_part = parts_mask.shape[0]
        # parts_pts = torch.zeros((num_part, 1, 3)).float()  # (num_parts, npts, 3)
        parts_box = torch.tensor(boxes).float()  # 2 * 12  # training using gt box as input
        gt_box = torch.tensor(boxes).float()  # gt box
        edge_pair = torch.tensor([id1, id2]).int()  # for fetching pair part id

        return {
            'obj_name': f'{obj_name}_pair{id1}_{id2}',
            'img': img,
            'mask': parts_mask,
            'box': parts_box,
            'gt_box': gt_box,
            # 'pts': parts_pts,
            'ids': edge_pair,
        }

    def __len__(self):
        return len(self.pair_list)


def build_data_loader(cfg, mode="train"):
    if mode == "train":
        dataset = PairPartDataset(
            pair_list_file=cfg.DATA.TOUCHPAIR.TRAIN.PAIR_LIST,
            mask_prefix=cfg.DATA.TOUCHPAIR.TRAIN.MASK_PREFIX,
            img_height=cfg.DATA.TOUCHPAIR.TRAIN.IMG_HEIGHT,
            img_width=cfg.DATA.TOUCHPAIR.TRAIN.IMG_WIDTH,
            dataset_name="train",
        )
    elif mode == "val":
        dataset = PairPartDataset(
            pair_list_file=cfg.DATA.TOUCHPAIR.VAL.PAIR_LIST,
            mask_prefix=cfg.DATA.TOUCHPAIR.VAL.MASK_PREFIX,
            img_height=cfg.DATA.TOUCHPAIR.VAL.IMG_HEIGHT,
            img_width=cfg.DATA.TOUCHPAIR.VAL.IMG_WIDTH,
            dataset_name="val",
        )
    elif mode == "test":
        dataset = PairPartDataset(
            pair_list_file=cfg.DATA.TOUCHPAIR.TEST.PAIR_LIST,
            mask_prefix=cfg.DATA.TOUCHPAIR.TEST.MASK_PREFIX,
            img_height=cfg.DATA.TOUCHPAIR.TEST.IMG_HEIGHT,
            img_width=cfg.DATA.TOUCHPAIR.TEST.IMG_WIDTH,
            dataset_name="test",
        )
    else:
        raise ValueError("Unknown mode: {}.".format(mode))

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
