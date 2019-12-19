import pandas as pd
import numpy as np
import os, shutil
from tqdm import tqdm

import sys
from imageio import imread
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import chain
import itertools
import logging
import warnings

from torch.nn import Sigmoid
from torch import Tensor, LongTensor
from torch.autograd import Variable

import torchvision.transforms as transforms

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch import optim
import matplotlib.pyplot as plt
import math
import torchvision.utils as vutils
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, _resnet, conv3x3, conv1x1, Bottleneck
import torch.nn as nn 
from torchvision.models.resnet import ResNet
import pickle
from attention import *
from dataset import *
from helper_functions import *
import time, datetime

class OurResnet(ResNet):
  def __init__(self, block, layers, out_channels, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
    super(OurResnet, self).__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group, 
                                    replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer)    
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    return x
   
def newResnet34(blk, pretrained=False, progress=True, out_channels=1, **kwargs):
  model = OurResnet(blk, [3,4,6,3], out_channels, **kwargs)
  return model

def oldResnet34(blk, pretrained=False, progress=True, out_channels=1, **kwargs):
  return _resnet('resnet34', blk, [3, 4, 6, 3], pretrained, progress, **kwargs)
  
