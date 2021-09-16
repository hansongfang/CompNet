from pathlib import Path

from CompNet.file_logger import file_logger_part, file_logger_binary_center, file_logger_relation


def get_test_output_dirname(cfg, mode):
    """Specify the test output dirname according to its loading file"""
    assert mode == 'test'
    if cfg.MODEL.CHOICE in ['RotNet', 'AxisLengthNet']:
        obj_test_file = cfg.DATA.PART.TEST.SHAPE_LIST
    elif cfg.MODEL.CHOICE == 'GroupAxisLengthNet':
        obj_test_file = cfg.DATA.NParts.TEST.PAIR_LIST
    elif cfg.MODEL.CHOICE == 'JointNet':
        obj_test_file = cfg.DATA.TOUCHPAIR.TEST.PAIR_LIST
    elif cfg.MODEL.CHOICE in ['ADJNet', 'SizeRelationNet']:
        obj_test_file = cfg.DATA.PartRelation.TEST.PAIR_LIST
    else:
        raise NotImplementedError()
    output_dir = f'{Path(obj_test_file).stem}'
    return output_dir


def build_file_logger(cfg, mode):
    assert mode == 'test'
    if cfg.MODEL.CHOICE in ['RotNet', 'AxisLengthNet']:
        return file_logger_part
    elif cfg.MODEL.CHOICE == 'GroupAxisLengthNet':
        return file_logger_part
    elif cfg.MODEL.CHOICE in ['JointNet']:
        return file_logger_binary_center
    elif cfg.MODEL.CHOICE in ['ADJNet', 'SizeRelationNet']:
        return file_logger_relation
    else:
        raise ValueError(f'Not supported model choice {cfg.MODEL.CHOICE}')
