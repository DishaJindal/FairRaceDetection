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

class CancerDataset(torch.utils.data.Dataset):
  def __init__(self, dataFolder, dataList):
    self.data = dataList
    self.dataFolder = dataFolder
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
    filename = self.data.iloc[index].filename
    if 'FullRes' in filename :
        filename = filename[8:]
    data = imread(os.path.join(self.dataFolder, filename))
    data = Image.fromarray(data)
    data = data.resize((512,512))
    data = np.array(data)
    data = data/256.0 
    data = torchvision.transforms.ToTensor()(data)
    data = data.float()
    label = self.data.iloc[index].label
    bmi = self.data.iloc[index].BMI
    return data, np.array([label, bmi]).astype(float)

df = pd.read_csv('LabelFile_2000.csv')
df['BMI'] = (df['BMI'] - df['BMI'].min())/(df['BMI'].max() - df['BMI'].min())
trainData = df[df.train == False]
validData = df[df.train == True]

dataFolder = "/cbica/home/santhosr/CBIG/BIRAD/Data"
vd = CancerDataset(dataFolder, validData) 
with open('/cbica/home/thodupuv/acv/data/Cancer/val_full_512.pk', 'wb') as f:
  pickle.dump(vd.repo, f)

td = CancerDataset(dataFolder, trainData)
with open('/cbica/home/thodupuv/acv/data/Cancer/train_full_512.pk', 'wb') as f:
  pickle.dump(td.repo, f)
