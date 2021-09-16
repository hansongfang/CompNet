from yacs.config import CfgNode as CN
from yacs.config import load_cfg


_C = CN()

# if set to @, the filename of config will be used by default
_C.OUTPUT_DIR = "@"
# Automatically resume weights from last checkpoints
_C.AUTO_RESUME = True
# For reproducibility...but not really because modern fast GPU libraries use
# non-deterministic op implementations
# -1 means not to set explicitly.
_C.RNG_SEED = 1

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

_C.DATA = CN()

_C.DATA.NUM_WORKERS = 1
_C.DATA.H5PY = False
_C.DATA.Version = 1

#-----------------Version1-----------------#

_C.DATA.PART = CN()
_C.DATA.PART.TRAIN = CN()
_C.DATA.PART.TRAIN.IMG_DIR = ""
_C.DATA.PART.TRAIN.PART_LIST = ""
_C.DATA.PART.TRAIN.IMG_HEIGHT = 256
_C.DATA.PART.TRAIN.IMG_WIDTH = 256
_C.DATA.PART.TRAIN.MASK_PREFIX = 'partmask'

_C.DATA.PART.VAL = CN()
_C.DATA.PART.VAL.IMG_DIR = ""
_C.DATA.PART.VAL.PART_LIST = ""
_C.DATA.PART.VAL.IMG_HEIGHT = 256
_C.DATA.PART.VAL.IMG_WIDTH = 256
_C.DATA.PART.VAL.MASK_PREFIX = 'partmask'

_C.DATA.PART.TEST = CN()
_C.DATA.PART.TEST.IMG_DIR = ""
_C.DATA.PART.TEST.PART_LIST = ""
_C.DATA.PART.TEST.SHAPE_LIST = ""
_C.DATA.PART.TEST.IMG_HEIGHT = 256
_C.DATA.PART.TEST.IMG_WIDTH = 256
_C.DATA.PART.TEST.MASK_PREFIX = 'partmask'

#-----------Import N parts-----------------#
_C.DATA.NParts = CN()
_C.DATA.NParts.TRAIN = CN()
_C.DATA.NParts.TRAIN.RENDER_DIR = ""
_C.DATA.NParts.TRAIN.PAIR_LIST = ""
_C.DATA.NParts.TRAIN.MASK_PREFIX = "partmask"
_C.DATA.NParts.TRAIN.IMG_HEIGHT = 256
_C.DATA.NParts.TRAIN.IMG_WIDTH = 256

_C.DATA.NParts.VAL = CN()
_C.DATA.NParts.VAL.RENDER_DIR = ""
_C.DATA.NParts.VAL.PAIR_LIST = ""
_C.DATA.NParts.VAL.MASK_PREFIX = "partmask"
_C.DATA.NParts.VAL.IMG_HEIGHT = 256
_C.DATA.NParts.VAL.IMG_WIDTH = 256

_C.DATA.NParts.TEST = CN()
_C.DATA.NParts.TEST.RENDER_DIR = ""
_C.DATA.NParts.TEST.PAIR_LIST = ""
_C.DATA.NParts.TEST.MASK_PREFIX = "partmask"
_C.DATA.NParts.TEST.IMG_HEIGHT = 256
_C.DATA.NParts.TEST.IMG_WIDTH = 256

#-----------Import two parts-----------------#
_C.DATA.TOUCHPAIR = CN()
_C.DATA.TOUCHPAIR.TRAIN = CN()
_C.DATA.TOUCHPAIR.TRAIN.PAIR_LIST = ""
_C.DATA.TOUCHPAIR.TRAIN.IMG_HEIGHT = 256
_C.DATA.TOUCHPAIR.TRAIN.IMG_WIDTH = 256
_C.DATA.TOUCHPAIR.TRAIN.MASK_PREFIX = 'partmask'

_C.DATA.TOUCHPAIR.VAL = CN()
_C.DATA.TOUCHPAIR.VAL.PAIR_LIST = ""
_C.DATA.TOUCHPAIR.VAL.IMG_HEIGHT = 256
_C.DATA.TOUCHPAIR.VAL.IMG_WIDTH = 256
_C.DATA.TOUCHPAIR.VAL.MASK_PREFIX = 'partmask'

_C.DATA.TOUCHPAIR.TEST = CN()
_C.DATA.TOUCHPAIR.TEST.PAIR_LIST = ""
_C.DATA.TOUCHPAIR.TEST.IMG_HEIGHT = 256
_C.DATA.TOUCHPAIR.TEST.IMG_WIDTH = 256
_C.DATA.TOUCHPAIR.TEST.MASK_PREFIX = 'partmask'

#-----------Import two parts relationship-----------------#
_C.DATA.PartRelation = CN()
_C.DATA.PartRelation.TRAIN = CN()
_C.DATA.PartRelation.TRAIN.IMG_DIR = ''
_C.DATA.PartRelation.TRAIN.PAIR_LIST = ''
_C.DATA.PartRelation.TRAIN.MASK_PREFIX = 'partmask'
_C.DATA.PartRelation.TRAIN.IMG_HEIGHT = 256
_C.DATA.PartRelation.TRAIN.IMG_WIDTH = 256
_C.DATA.PartRelation.TRAIN.USE_WEIGHT = False

_C.DATA.PartRelation.VAL = CN()
_C.DATA.PartRelation.VAL.IMG_DIR = ''
_C.DATA.PartRelation.VAL.PAIR_LIST = ''
_C.DATA.PartRelation.VAL.MASK_PREFIX = 'partmask'
_C.DATA.PartRelation.VAL.IMG_HEIGHT = 256
_C.DATA.PartRelation.VAL.IMG_WIDTH = 256
_C.DATA.PartRelation.VAL.USE_WEIGHT = False

_C.DATA.PartRelation.TEST = CN()
_C.DATA.PartRelation.TEST.IMG_DIR = ''
_C.DATA.PartRelation.TEST.PAIR_LIST = ''
_C.DATA.PartRelation.TEST.MASK_PREFIX = 'partmask'
_C.DATA.PartRelation.TEST.IMG_HEIGHT = 256
_C.DATA.PartRelation.TEST.IMG_WIDTH = 256
_C.DATA.PartRelation.TEST.USE_WEIGHT = False

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.WEIGHT = ""
_C.MODEL.EDGE_CHANNELS = ()
_C.MODEL.CHOICE = 'RotNet'

_C.MODEL.ROTNET = CN()
_C.MODEL.ROTNET.IMG_OPTION = "conca"
_C.MODEL.ROTNET.NET_OPTION = 'RotSize'
_C.MODEL.ROTNET.RELATIVE_SIZE = True
_C.MODEL.ROTNET.POOLING = False
_C.MODEL.ROTNET.ORTHONORMAL = True
_C.MODEL.ROTNET.LOSS = "RotGTSize"
_C.MODEL.ROTNET.METRIC = "RotGTSize"

_C.MODEL.JOINTNET = CN()
_C.MODEL.JOINTNET.NORM_SCALE = False
_C.MODEL.JOINTNET.MASK_IMAGE = False
_C.MODEL.JOINTNET.LOSS = "ChamferDist"
_C.MODEL.JOINTNET.METRIC = "ChamferDist"

_C.MODEL.AxisLengthNet = CN()
_C.MODEL.AxisLengthNet.LOSS = 'L2'
_C.MODEL.AxisLengthNet.METRIC = 'ChamferDist'

_C.MODEL.GroupAxisLengthNet = CN()
_C.MODEL.GroupAxisLengthNet.LOSS = 'L2'
_C.MODEL.GroupAxisLengthNet.METRIC = 'ChamferDist'

# ---------------------------------------------------------------------------- #
# Solver (optimizer)
# ---------------------------------------------------------------------------- #

_C.SOLVER = CN()

# Type of optimizer
_C.SOLVER.TYPE = "Adam"

# Basic parameters of solvers
# Notice to change learning rate according to batch size
_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.WEIGHT_DECAY = 0.0

# Specific parameters of solvers
_C.SOLVER.RMSprop = CN()
_C.SOLVER.RMSprop.alpha = 0.9

_C.SOLVER.SGD = CN()
_C.SOLVER.SGD.momentum = 0.9

_C.SOLVER.Adam = CN()
_C.SOLVER.Adam.weight_decay = 0.0

# ---------------------------------------------------------------------------- #
# Scheduler (learning rate schedule)
# ---------------------------------------------------------------------------- #
_C.SCHEDULER = CN()
_C.SCHEDULER.TYPE = "StepLR"

_C.SCHEDULER.MAX_EPOCH = 2

_C.SCHEDULER.StepLR = CN()
_C.SCHEDULER.StepLR.step_size = 5
_C.SCHEDULER.StepLR.gamma = 0.9

_C.SCHEDULER.MultiStepLR = CN()
_C.SCHEDULER.MultiStepLR.milestones = ()
_C.SCHEDULER.MultiStepLR.gamma = 0.1

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 1


# The period to save a checkpoint
_C.TRAIN.CHECKPOINT_PERIOD = 1000
_C.TRAIN.LOG_PERIOD = 10
# The period to validate
_C.TRAIN.VAL_PERIOD = 0
# Data augmentation. The format is "method" or ("method", *args)
# For example, ("PointCloudRotate", ("PointCloudRotatePerturbation",0.1, 0.2))
_C.TRAIN.AUGMENTATION = ()

# Regex patterns of modules and/or parameters to freeze
# For example, ("bn",) will freeze all batch normalization layers' weight and bias;
# And ("module:bn",) will freeze all batch normalization layers' running mean and var.
_C.TRAIN.FROZEN_PATTERNS = []
_C.TRAIN.FROZEN_PARAMS = ''
_C.TRAIN.FROZEN_MODULES = []

_C.TRAIN.VAL_METRIC = "l1"

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

_C.TEST.BATCH_SIZE = 1

# The path of weights to be tested. "@" has similar syntax as OUTPUT_DIR.
# If not set, the last checkpoint will be used by default.
_C.TEST.WEIGHT = ""

# Data augmentation.
_C.TEST.AUGMENTATION = ()

_C.TEST.LOG_PERIOD = 10

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.VAL = CN()

_C.VAL.BATCH_SIZE = 1


def load_cfg_from_file(cfg_filename):
    """Load config from a file

    Args:
        cfg_filename (str):

    Returns:
        CfgNode: loaded configuration

    """
    with open(cfg_filename, "r") as f:
        cfg = load_cfg(f)

    cfg_template = _C
    cfg_template.merge_from_other_cfg(cfg)
    return cfg_template


def get_cur_cfg():
    return _C
