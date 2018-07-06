from torchloop.util.torchloop_logging
from easydict import EasyDict as edict
import torch
logger = torchloop_logging.tl_logger()

class torch_device:
    def __init__(self, **named_param):
        conf = edict(named_param)
        # right now only device name is supported
        # TODO: distributed environment
        self.device = conf.device_name

    def decorate_model(self, nn_o):
        if self.device == "cuda":
            return nn_o.cuda()
        else:
            return nn_o
    
    def init_tensor_dtype(self):
        if self.device == "cuda":
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # TODO: also set when tensor_type when device is cpu

