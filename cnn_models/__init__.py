import random
import os
import numpy as np
import torch

from .alexnet import alexnet
from .vggnet import vgg
from .zfnet import zfnet
from .shufflenet import shufflenet_v2_x0_5, shufflenet_v2_x1_0

cfgs = {
    'alexnet': alexnet,
    'zfnet': zfnet,
    'vgg11': vgg(model_name='vgg11'),
    'vgg13': vgg(model_name='vgg13'),
    'vgg': vgg(model_name='vgg16'),
    'vgg19': vgg(model_name='vgg19'),
    'shufflenet_0.5':shufflenet_v2_x0_5,
    'shufflenet': shufflenet_v2_x1_0,
}

def find_model_using_name(model_name, num_classes):  
    return cfgs[model_name](num_classes)

 