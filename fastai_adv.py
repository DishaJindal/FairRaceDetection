import pandas as pd
import numpy as np
import os, shutil
from tqdm import tqdm
from sklearn.metrics import accuracy_score

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
from torchvision.models.resnet import ResNet, resnet50, resnet18
import pickle


from fastai.script import *
from fastai.vision import *
from fastai.distributed import *
from fastai.callbacks import *
from fastai.callback import *

#### Data set

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

### Models

def print_grad(model_part):
    for name, child in model_part.named_children():
        for name2, params in child.named_parameters():
            print(name2, params)

def freeze_part(model_part):
    for param in model_part.parameters():
        param.requires_grad = False

def un_freeze_part(model_part):
    for param in model_part.parameters():
        param.requires_grad = True

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
   
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class fastAIModel(nn.Module):
    def __init__(self):
        super(fastAIModel, self).__init__()
        self.resnet = nn.Sequential(*list(resnet50().children())[:-2])
        self.seq1 = nn.Sequential(
            AdaptiveConcatPool2d(), Flatten(),
            nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=4096, out_features=512, bias=True),
            nn.ReLU(inplace = True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=2, bias=True))
        
        self.seq2 = nn.Sequential(
            AdaptiveConcatPool2d(), Flatten(),
            nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=4096, out_features=512, bias=True),
            nn.ReLU(inplace = True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=4, bias=True))

        self.classifier = nn.Sequential(self.resnet, self.seq1)
        self.flag = False

    def load_classifier(self, filename):
        self.classifier.load_state_dict(torch.load(filename)['model'])

    def change_model_state(self, mode):
        if mode == 'classification':
            un_freeze_part(self.seq1)
            un_freeze_part(self.resnet)
            freeze_part(self.seq2)
            self.flag = False
        else:
            freeze_part(self.seq1)
            freeze_part(self.resnet)
            un_freeze_part(self.seq2)
            self.flag = True

       
    def forward(self, x):
        if self.flag:
            self.resnet.eval()
            self.seq1.eval()

        else:
            self.seq2.eval()

        x = self.resnet(x)
        y1 = self.seq1(x)
        y2 = self.seq2(x)
        return torch.cat([y1, y2], dim=1)



## Loss Functions
def classification_loss_function(out, target):
    new_target = target[:,0].long()
    return F.cross_entropy(out[:,:2], new_target) 

def adv_loss_function(out, target):
    new_target = target[:,1].long()
    return F.cross_entropy(out[:,2:], new_target)

def combined_loss_function(out, target):
    closs = classification_loss_function(out, target)
    rloss = adv_loss_function(out, target)
    return closs - 2* rloss

class MetricCall(Callback):
    def __init__(self, func):
        super().__init__()
        self.output = None
        self.target = None
        self.func = func

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0.,0

    def on_batch_end(self, last_output, last_target, **kwargs):
        if self.output is None:
            self.output = last_output.detach().cpu().clone()
        else:
            self.output = torch.cat([self.output, last_output.detach().cpu().clone()], dim=0)

        if self.target is None:
            self.target = last_target.detach().cpu().clone()
        else:
            self.target = torch.cat([self.target, last_target.detach().cpu().clone()], dim=0)

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics,  self.func(self.output, self.target))



# class AverageMetric(Callback):
#     def __init__(self):
#         super().__init__()
#         self.count = 0
#         self.val = 0

#     def on_epoch_begin(self, **kwargs):
#         "Set the inner value to 0."
#         self.val, self.count = 0.,0

#     def on_batch_end(self, last_output, last_target, **kwargs):
#         self.count += last_target[0].size(0)
#         self.val += accuracy(last_output, last_target)*last_target[0].size(0)

#     def on_epoch_end(self, last_metrics, **kwargs):
#         return add_metrics(last_metrics, self.val / self.count)
 

def accuracy(input:Tensor, targs:Tensor)->Rank0Tensor:
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input[:,:2].argmax(dim=-1).view(n,-1).cpu()
    targs = targs[:,0].view(n,-1).long().cpu()
    return (input==targs).float().mean().item()

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

def bias(input:Tensor, targetFull:Tensor)->Rank0Tensor:
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    n = targetFull.shape[0]
    pred = input[:,:2].argmax(dim=-1).view(n,-1)
    targs = targetFull[:,0].view(n,-1).long()
    bmi = targetFull[:,2].view(n,-1)
    pred_label = list(pred.cpu().view(-1).numpy())
    true_label = list(targs.cpu().view(-1).numpy())
    # pred_bmi = list((input[:,2].cpu().view(-1).numpy()) * 67.7 + 4.7)
    pred_bmi = list((bmi.cpu().view(-1).numpy()) * 67.7 + 4.7)
    return calculate_bias(pred_label, true_label, pred_bmi)

## 
class AdvCallback(LearnerCallback):
    def __init__(self, learn:Learner, model, databunch, ep_limit, lr, monitor:str='adv_call_back', mode:str='auto'):
        super().__init__(learn)#, monitor=monitor, mode=mode)
        self.model = model
        self.databunch = databunch
        self.ep_limit = ep_limit
        self.lr = lr
        # super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        train_adv(self.ep_limit, self.model, self.databunch, self.lr)


## Train Functions

def train_class(epoch, model, databunch, lr):
    model.change_model_state('classification')
    l = Learner(databunch, model, metrics=[MetricCall(accuracy), MetricCall(bias)], loss_func=classification_loss_function)
    l.fit_one_cycle(epoch, lr)


def train_adv(epoch, model, databunch, lr):
    model.change_model_state('adv')
    l = Learner(databunch, model, metrics=[MetricCall(accuracy), MetricCall(bias)], loss_func=adv_loss_function)
    l.fit_one_cycle(epoch, lr)
    model.change_model_state('classification')

# def train_class(epoch, model, databunch, lr, adv_lr, adv_epoch):
#     model.change_model_state('classification')
#     l = Learner(databunch, model, metrics=[AverageMetric(accuracy), MetricCall(bias)], loss_func=combined_loss_function)
#     # adv_call_back = AdvCallback(l, model, databunch2, adv_epoch , adv_lr)
#     l.fit_one_cycle(epoch, lr)

## Create Model
faimodel = fastAIModel()

faimodel.load_classifier('/cbica/home/santhosr/CBIG/BIRAD/models/model_resnet50_id10_acc833_loss386.pth')


train_dataset = CancerDatasetNP('/cbica/home/thodupuv/acv/data/Cancer/train_full_256.pk') 
test_dataset = CancerDatasetNP('/cbica/home/thodupuv/acv/data/Cancer/val_full_256.pk') 

data1 = DataBunch.create(train_dataset, test_dataset)
data2 = DataBunch.create(train_dataset, test_dataset)


## Pretrain
# pretrain_class(20, faimodel, data1, 1e-10)

# test(faimodel, data1)
print("######################################################")
print("Pre Training CLS")
train_class(7, faimodel, data1, 1e-5)
print("######################################################")

print("######################################################")
print("Pre Training ADV")
train_adv(20, faimodel, data1, 1e-4)
print("######################################################")

faimodel.change_model_state('classification')
l1 = Learner(data1, faimodel, metrics=[MetricCall(accuracy), MetricCall(bias)], loss_func=combined_loss_function)
l2 = Learner(data1, faimodel, metrics=[MetricCall(accuracy), MetricCall(bias)], loss_func=adv_loss_function)

## Train
for i in range(50):
    print("######################################################")
    print("Training CLS")
    faimodel.change_model_state('classification')
    if i == 0: 
        l1.fit(1, 1e-5)
    else:
        l1.fit(1, l1.opt.lr, l1.opt.wd)
    print("######################################################")
    print("######################################################")
    print("Training adv")
    faimodel.change_model_state('adv') 
    if i == 0:   
        l2.fit(1, 1e-5)
    else:
        l2.fit(1, l2.opt.lr, l2.opt.wd)
    print("######################################################")




