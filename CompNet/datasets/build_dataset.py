from CompNet.datasets.dataset_2parts import build_data_loader as build_2parts_data
from CompNet.datasets.dataset_part import build_data_loader as build_part_data
from CompNet.datasets.dataset_relation import build_data_loader as build_relation_data
from CompNet.datasets.dataset_npart import build_data_loader as build_npart_data


def build_data_loader(cfg, mode):
    if cfg.MODEL.CHOICE in ['RotNet', 'AxisLengthNet']:
        data_loader = build_part_data(cfg, mode)
    elif cfg.MODEL.CHOICE == 'GroupAxisLengthNet':
        if mode == 'test':
            data_loader = build_npart_data(cfg, mode)
        else:
            data_loader = build_2parts_data(cfg, mode)
    elif cfg.MODEL.CHOICE == 'JointNet':
        data_loader = build_2parts_data(cfg, mode)
    elif cfg.MODEL.CHOICE in ['ADJNet', 'SizeRelationNet']:
        data_loader = build_relation_data(cfg, mode)
    else:
        raise NotImplementedError()

    return data_loader
