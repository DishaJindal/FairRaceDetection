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

class CancerDatasetNP(torch.utils.data.Dataset):
  def __init__(self, file_path):
      self.c = 2
      self.is_empty = False
      with open(file_path, 'rb') as f:
          self.repo = pickle.load(f)

  def __getitem__(self, index):
    im, lab = self.repo[index]
    im = im[0, :, :]
    label = lab[0]
    norm_bmi = lab[1]
    bmi = lab[1] * 67.7 + 4.7

    bmi_bucket = -1
    if(bmi < 15):
        bmi_bucket = 0
    elif(bmi < 35):
        bmi_bucket = 1
    elif(bmi < 55):
        bmi_bucket = 2
    else:
        bmi_bucket = 3

    return np.stack([im,im, im]), np.array([label, bmi_bucket, norm_bmi]).astype(float)
    
  def __len__(self):
    return len(list(self.repo.keys()))
