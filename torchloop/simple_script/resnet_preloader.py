import torchvision
import torch.nn as nn
from torchloop.util import tl_logging
import torch
logger = tl_logging.tl_logger(tl_logging.DEBUG, True)

resnet = torchvision.models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

children = list(resnet.children())#[:23]
logger.debug("feathers type {}".format(type(children))) # list
logger.debug("feathers len {}".format(len(children))) # 10
logger.debug("feathers 1 {}".format(children[0]))
models = resnet._modules
logger.debug("resnet all layers are \n{}".format(
    models.keys()))

#####
# replace the last fc layer 
#####
# resnet.fc = nn.Linear(resnet.fc.in_features, 100)  
# 100 is an example.

images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
logger.info("output_shape {}".format(outputs.size()))

####
# figuring out the outputsize
####
input_layer = resnet.conv1

# torch.save(resnet, 'first_saved.ckpt') # 45Mb
model = torch.load('first_saved.ckpt')

logger.info("forward feeding")
outputs = model(images)
logger.info("output_shape {}".format(outputs.size()))

# test ok
