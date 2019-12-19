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


from fastai.script import *
from fastai.vision import *
from fastai.distributed import *

def classification_loss_function(out, target):
	new_target = target[:,0]
	return 


class CancerDatasetNP(torch.utils.data.Dataset):
  def __init__(self, file_path):
  	self.c = 2
  	with open(file_path, 'rb') as f:
  		self.repo = pickle.load(f)

  def __getitem__(self, index):
    im, lab = self.repo[index]
    im = im[0, :, :]
    return np.stack([im,im, im]), int(lab[0])
    
  def __len__(self):
    return len(list(self.repo.keys()))



train_dataset = CancerDatasetNP('/cbica/home/thodupuv/acv/data/Cancer/train_full_256.pk') 
test_dataset = CancerDatasetNP('/cbica/home/thodupuv/acv/data/Cancer/val_full_256.pk') 

data = DataBunch.create(train_dataset, test_dataset)
# print(data.loss_func)
learn = cnn_learner(data, models.resnet50, metrics=accuracy, loss_func=F.cross_entropy)
learn.fit_one_cycle(80, 1e-4)