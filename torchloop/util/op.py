from enum import Flag
from torchloop.util.config_object

class OP_T(Flag):
    TRAIN = auto()
    TEST = auto()
    INFER = auto()
    TRAIN_TEST = TRAIN | TEST
    TRAIN_INFER = TRAIN | INFER
    TRAIN_TEST_INFER = TRAIN | TEST | INFER

class op_t(config_object.configurable):
    pass
