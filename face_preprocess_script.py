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

class FairFaceDataset(torch.utils.data.Dataset):
  def __init__(self, dataFolder, dataList):
    self.data = dataList
    self.dataFolder = dataFolder
    self.genderMap = {"Male":0, "Female":1}
    self.raceMap = {"East Asian":0, "Indian":1, "Black":2,"Middle Eastern":3, "White":4,"Latino_Hispanic":5, "Southeast Asian":6}
    self.repo = {}
    self.pre_process()

  def pre_process(self):
    for i in range(len(self.data)):
      print(i)
      self.repo[i] = self.getitem(i)
      sys.stdout.flush()

  def __getitem__(self, index):
    return self.repo[index]

    
  def __len__(self):
    return len(self.data)
    
  def getitem(self, index):
    filename = self.data.iloc[index].file
    data = imread(os.path.join(self.dataFolder, filename))
    data = Image.fromarray(data)
    # data = data.resize((256, 256))
    data = np.array(data)
    data = data/256.0 
    data = torchvision.transforms.ToTensor()(data)
    data = data.float()
    gender = self.genderMap[self.data.iloc[index].gender]
    race = self.raceMap[self.data.iloc[index].race]
    return data, np.array([race, gender]).astype(float)


trainData = pd.read_csv('/cbica/home/thodupuv/acv/data/FairFace/fairface_label_train.csv')
validData = pd.read_csv('/cbica/home/thodupuv/acv/data/FairFace/fairface_label_val.csv')
dataFolder = '/cbica/home/thodupuv/acv/data/FairFace'

vd = FairFaceDataset(dataFolder, validData[:3000]) 
with open('/cbica/home/thodupuv/acv/data/FairFace/val_3k_224.pk', 'wb') as f:
  pickle.dump(vd.repo, f)

td = FairFaceDataset(dataFolder, trainData[:20000])
with open('/cbica/home/thodupuv/acv/data/FairFace/train_20k_224.pk', 'wb') as f:
  pickle.dump(td.repo, f)
