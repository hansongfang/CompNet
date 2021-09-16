"""
Dataloader for parts relationship prediction.
"""
from pathlib import Path
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


class PartRelationDataset(Dataset):

    def __init__(self,
                 render_dir,
                 pair_list_file,
                 mask_prefix='partmask',
                 img_height=256,
                 img_width=256,
                 use_weight=False,
                 dataset_name='train'):
        self.render_dir = Path(render_dir)
        tmp_file = str(self.render_dir/pair_list_file)
        self.pair_list = load_pickle(tmp_file)
        logger.info(f'Parts relation dataloader, {len(self.pair_list)} pairs')

        self.mask_prefix = mask_prefix
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.dataset_name = dataset_name
        self.use_weight = use_weight

    def __getitem__(self, index):
        render_dir = self.render_dir
        obj, id1, id2, label = self.pair_list[index]
        render_dir = Path(render_dir)
        obj_name = Path(obj).stem

        img_dir = render_dir / obj_name

        # loading images
        img_file = img_dir/'img.png'
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

        img = torch.tensor(imgs).permute(0, 3, 1, 2).float()  # 2, 3 * height * width
        parts_mask = torch.tensor(masks).float()  # 2 * height * width

        # !!! we use gt box for training, at testing ,we use prediction box from previous stage
        edge_pair = torch.tensor([id1, id2]).int()
        relation_label = torch.tensor([label]).float()

        if self.use_weight:
            if label == 1.0:
                relation_weight = torch.tensor([10.0]).float()
            else:
                relation_weight = torch.tensor([1.0]).float()
        else:
            relation_weight = torch.tensor([1.0]).float()

        return {
            'obj_name': f'{obj_name}_pair{id1}_{id2}',
            'img': img,
            'mask': parts_mask,
            'adj': relation_label,
            'ids': edge_pair,
            'weight': relation_weight,
        }

    def __len__(self):
        return len(self.pair_list)


def build_data_loader(cfg, mode="train"):
    if mode == "train":
        dataset = PartRelationDataset(
            render_dir=cfg.DATA.PartRelation.TRAIN.IMG_DIR,
            pair_list_file=cfg.DATA.PartRelation.TRAIN.PAIR_LIST,
            mask_prefix=cfg.DATA.PartRelation.TRAIN.MASK_PREFIX,
            img_height=cfg.DATA.PartRelation.TRAIN.IMG_HEIGHT,
            img_width=cfg.DATA.PartRelation.TRAIN.IMG_WIDTH,
            use_weight=cfg.DATA.PartRelation.TRAIN.USE_WEIGHT,
            dataset_name="train",
        )
    elif mode == "val":
        dataset = PartRelationDataset(
            render_dir=cfg.DATA.PartRelation.VAL.IMG_DIR,
            pair_list_file=cfg.DATA.PartRelation.VAL.PAIR_LIST,
            mask_prefix=cfg.DATA.PartRelation.VAL.MASK_PREFIX,
            img_height=cfg.DATA.PartRelation.VAL.IMG_HEIGHT,
            img_width=cfg.DATA.PartRelation.VAL.IMG_WIDTH,
            use_weight=cfg.DATA.PartRelation.VAL.USE_WEIGHT,
            dataset_name="val",
        )
    elif mode == "test":
        dataset = PartRelationDataset(
            render_dir=cfg.DATA.PartRelation.TEST.IMG_DIR,
            pair_list_file=cfg.DATA.PartRelation.TEST.PAIR_LIST,
            mask_prefix=cfg.DATA.PartRelation.TEST.MASK_PREFIX,
            img_height=cfg.DATA.PartRelation.TEST.IMG_HEIGHT,
            img_width=cfg.DATA.PartRelation.TEST.IMG_WIDTH,
            use_weight=cfg.DATA.PartRelation.TEST.USE_WEIGHT,
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
