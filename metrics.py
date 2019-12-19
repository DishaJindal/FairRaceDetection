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
from backbone import *
import time, datetime

# Helper function to calculate bias
def calculate_bias(pred_list, true_list, bmi_list):
    d = {'BMI': bmi_list, 'truthLabel': true_list, 'predLabel': pred_list}
    predDf = pd.DataFrame(d, columns=['BMI','truthLabel','predLabel'])

    outputData = []
    for i in range(8):
      bmi_min,bmi_max = 15 + 5*i, 15 + 5*(i+1)
      subDf = predDf[predDf.BMI.between(bmi_min,bmi_max)]     
      whiteDf = subDf[subDf.truthLabel==1]
      aaDf = subDf[subDf.truthLabel==0]
         
      tempDf = pd.concat([whiteDf, aaDf])
     
      whiteAcc = np.round(accuracy_score(whiteDf.truthLabel, whiteDf.predLabel),2)
      aaAcc = np.round(accuracy_score(aaDf.truthLabel, aaDf.predLabel),2)
     
      acc = np.round(accuracy_score(tempDf.truthLabel,tempDf.predLabel),2)
      outputData.append([bmi_min, bmi_max, len(subDf), acc, whiteAcc,aaAcc])
       
    tempDf = pd.DataFrame(outputData)
    tempDf.columns = ['BMI_min','BMI_max','numImages','Total Acc','White Acc','Black Acc']
    totalDiff = 0
    totalValidBuckets = 0
    for i in range(len(tempDf)):
        diff = np.abs(tempDf.iloc[i]['White Acc'] - tempDf.iloc[i]['Black Acc'])     
        if pd.isna(diff) == False:            
            totalValidBuckets += 1
            totalDiff += diff

    biasScore = totalDiff/totalValidBuckets
    return torch.tensor(biasScore) 

# Main function to calculate bias
def bias(input:Tensor, targetFull:Tensor)->Rank0Tensor:
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    n = targetFull.shape[0]
    pred = input[:,:2].argmax(dim=-1).view(n,-1)
    targs = targetFull[:,0].view(n,-1).long()
    bmi = targetFull[:,2].view(n,-1)
    pred_label = list(pred.cpu().view(-1).numpy())
    true_label = list(targs.cpu().view(-1).numpy())
    pred_bmi = list((bmi.cpu().view(-1).numpy()) * 67.7 + 4.7)
    return calculate_bias(pred_label, true_label, pred_bmi)