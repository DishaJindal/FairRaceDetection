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

def get_norm(norm_type, num_features, num_groups=32, eps=1e-5):
  if norm_type == 'BatchNorm':
    return nn.BatchNorm2d(num_features, eps=eps)
  elif norm_type == "GroupNorm":
    return nn.GroupNorm(num_groups, num_features, eps=eps)
  elif norm_type == "InstanceNorm":
    return nn.InstanceNorm2d(num_features, eps=eps, affine=True, track_running_stats=True)
  else:
    raise Exception('Unknown Norm Function : {}'.format(norm_type))

class Conv2dNormRelu(nn.Module):
  def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=True, norm_type='Unknown'):
    super(Conv2dNormRelu, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias),
        get_norm(norm_type, out_ch),
        nn.ReLU(inplace=True))
  def forward(self, x):
    return self.conv(x)

class CAModule(nn.Module):
  def __init__(self, num_channels, reduc_ratio=2):
    super(CAModule, self).__init__()
    self.num_channels = num_channels
    self.reduc_ratio = reduc_ratio
    self.fc1 = nn.Linear(num_channels, num_channels // reduc_ratio, bias=True)
    self.fc2 = nn.Linear(num_channels // reduc_ratio, num_channels, bias=True)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
  def forward(self, feat_map):
    gap_out = feat_map.view(feat_map.size()[0], self.num_channels, -1).mean(dim=2)
    fc1_out = self.relu(self.fc1(gap_out))
    fc2_out = self.sigmoid(self.fc2(fc1_out))
    fc2_out = fc2_out.view(fc2_out.size()[0], fc2_out.size()[1], 1, 1)
    feat_map = torch.mul(feat_map, fc2_out)
    return feat_map, [fc2_out]

class SAModule(nn.Module):
  def __init__(self, num_channels):
    super(SAModule, self).__init__()
    self.num_channels = num_channels
    self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels // 8, kernel_size=1)
    self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels // 8, kernel_size=1)
    self.conv3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)
    self.gamma = nn.Parameter(torch.zeros(1))
  
  def forward(self, feat_map):
    batch_size, num_channels, height, width = feat_map.size()
    conv1_proj = self.conv1(feat_map).view(batch_size, -1, width * height).permute(0, 2, 1)
    conv2_proj = self.conv2(feat_map).view(batch_size, -1, width * height)
    relation_map = torch.bmm(conv1_proj, conv2_proj)
    attention = F.softmax(relation_map, dim=-1)
    conv3_proj = self.conv3(feat_map).view(batch_size, -1, width * height)
    feat_refine = torch.bmm(conv3_proj, attention.permute(0, 2, 1))
    feat_refine = feat_refine.view(batch_size, num_channels, height, width)
    feat_map = self.gamma * feat_refine + feat_map
    return feat_refine, [self.gamma * feat_refine, feat_refine]

class FPAModule(nn.Module):
  def __init__(self, num_channels, norm_type):
    super(FPAModule, self).__init__()
    self.gap_branch = nn.Sequential(nn.AdaptiveAvgPool2d(1), Conv2dNormRelu(num_channels, num_channels, kernel_size=1, norm_type=norm_type))
    self.mid_branch = Conv2dNormRelu(num_channels, num_channels, kernel_size=1, norm_type=norm_type)
    self.downsample1 = Conv2dNormRelu(num_channels, 1, kernel_size=3, stride=2, padding=1, norm_type=norm_type)
    self.downsample2 = Conv2dNormRelu(1, 1, kernel_size=3, stride=2, padding=1, norm_type=norm_type)
    self.scale1 = Conv2dNormRelu(1, 1, kernel_size=3, padding=1, norm_type=norm_type)
    self.scale2 = Conv2dNormRelu(1, 1, kernel_size=3, padding=1, norm_type=norm_type)
  
  def forward(self, feat_map):
    height, width = feat_map.size(2), feat_map.size(3)
    gap_branch = self.gap_branch(feat_map)
    gap_branch = nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)(gap_branch)
    mid_branch = self.mid_branch(feat_map)
    scale1 = self.downsample1(feat_map)
    scale2 = self.downsample2(scale1)
    scale2 = self.scale2(scale2)
    scale2 = nn.Upsample(size=(int(math.ceil(height*1.0 / 2.0)), int(math.ceil(width*1.0 / 2.0))), mode='bilinear', align_corners=False)(scale2)
    scale1 = self.scale1(scale1) + scale2
    scale1 = nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)(scale1)
    feat_map = torch.mul(scale1, mid_branch) + gap_branch
    return feat_map, []

class AttMap(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(AttMap, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.sigmoid(out)
        return out, [out]

class AttMap2(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(AttMap2, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.sigmoid(out)
        return out, [out]
