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


exp_name = 'image_exp'
exp_folder = '/cbica/home/thodupuv/acv/models/Cancer/' + exp_name

try:
  os.mkdir(exp_folder)
except:
  print('Folder already exist')



import subprocess
def printNvidiaSmi():
  sp = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  out_str = sp.communicate()
  out_list = out_str[0].decode("utf-8").split('\n')

  for i in out_list:
    print(i)


warnings.filterwarnings('ignore')

import time, datetime

def delete_logs():
  folder = "/cbica/home/thodupuv/acv/logs"
  for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))

#delete_logs()
def get_tensorboard_logger():
  global exp_name
  logs_base_dir = "/cbica/home/thodupuv/acv/logs"
  logs_dir = logs_base_dir + "/run_" + exp_name + "_" + str(time.mktime(datetime.datetime.now().timetuple()))
  os.makedirs(logs_dir, exist_ok=True)
  from torch.utils.tensorboard import SummaryWriter
  # %tensorboard --logdir {logs_base_dir} --port=9002
  logger = SummaryWriter(logs_dir)
  return logger

"""# Datasets

## Cancer Dataset
"""

class CancerDatasetNP(torch.utils.data.Dataset):
  def __init__(self, file_path):
    with open(file_path, 'rb') as f:
      self.repo = pickle.load(f)

  def __getitem__(self, index):
    im, lab = self.repo[index]
    return im, lab
    
  def __len__(self):
    return len(list(self.repo.keys()))


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
    # self.downsample3 = Conv2dNormRelu(1, 1, kernel_size=3, stride=2, padding=1, norm_type=norm_type)
    self.scale1 = Conv2dNormRelu(1, 1, kernel_size=3, padding=1, norm_type=norm_type)
    self.scale2 = Conv2dNormRelu(1, 1, kernel_size=3, padding=1, norm_type=norm_type)
    # self.scale3 = Conv2dNormRelu(1, 1, kernel_size=3, padding=1, norm_type=norm_type)
  
  def forward(self, feat_map):
    height, width = feat_map.size(2), feat_map.size(3)
    gap_branch = self.gap_branch(feat_map)
    gap_branch = nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)(gap_branch)
    mid_branch = self.mid_branch(feat_map)
    scale1 = self.downsample1(feat_map)
    scale2 = self.downsample2(scale1)
    # scale3 = self.downsample3(scale2)
    # scale3 = self.scale3(scale3)
    # scale3 = nn.Upsample(size=(int(math.ceil(height*1.0 / 4.0)), int(math.ceil(width*1.0 / 4.0))), mode='bilinear', align_corners=False)(scale3)
    # scale2 = self.scale2(scale2) + scale3
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
        # self.conv3 = conv1x1(planes,1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.conv3(out)
        out = torch.sigmoid(out)
        return out, [out]


class AttMap2(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(AttMap2, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        # self.conv3 = conv1x1(planes,1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.conv3(out)
        out = torch.sigmoid(out)
        return out, [out]






"""## ResNet Wrapper"""

global_batchid = -1
global_epoch = -1
global_id = 0
global_logger = get_tensorboard_logger()

class AdvancedBasicBlock(BasicBlock):
  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
    super(AdvancedBasicBlock, self).__init__(inplanes, planes, stride=stride, downsample=downsample, groups=groups,
                 base_width=base_width, dilation=dilation, norm_layer=norm_layer)
    
    print(self.bn2.num_features)
    global global_id
    self.id = global_id
    global_id += 1
    self.attention_layer = None
  
  def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    
    out = self.conv2(out)
    out = self.bn2(out)

    attention_map, plot_data = self.attention_layer(x)
    plot_data.append(out)
    plot_data.append(out * attention_map)
    out = out * attention_map + out
    plot_data.append(out)
    global global_batchid, global_epoch, global_logger

    local_batch_id = global_batchid
    local_epoch = global_epoch

    if not self.training and local_batch_id < 1:
      for index, value in enumerate(plot_data):
        for channel_id in range(value.shape[1]):
          if(channel_id > 10):
            continue
          key = "_ListId_" + str(index) + "_Layer_" + str(self.id) + "_BatchId_" + str(local_batch_id) + "_ChannelId_" + str(channel_id)
          sample = value[:, channel_id, :, :].unsqueeze(1)
          # sample = sample.repeat(1, 3, 1, 1)
          #global_logger.add_images(key, vutils.make_grid(sample.cpu(), padding=2).unsqueeze(0),  local_epoch)
          
    
    if self.downsample is not None:
      identity = self.downsample(x)
    
    out += identity
    out = self.relu(out)
    return out

class AdvancedBasicBlock_CAM(AdvancedBasicBlock):
  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, 
               base_width=64, dilation=1, norm_layer=None):
    super(AdvancedBasicBlock_CAM, self).__init__(inplanes, planes, stride=stride, downsample=downsample, groups=groups,
                 base_width=base_width, dilation=dilation, norm_layer=norm_layer)
    self.attention_layer = CAModule(self.bn2.num_features)

class AdvancedBasicBlock_SAM(AdvancedBasicBlock):
  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, 
               base_width=64, dilation=1, norm_layer=None):
    super(AdvancedBasicBlock_SAM, self).__init__(inplanes, planes, stride=stride, downsample=downsample, groups=groups,
                 base_width=base_width, dilation=dilation, norm_layer=norm_layer)
    self.attention_layer = SAModule(self.bn2.num_features)

class AdvancedBasicBlock_FPA(AdvancedBasicBlock):
  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, 
               base_width=64, dilation=1, norm_layer=None):
    super(AdvancedBasicBlock_FPA, self).__init__(inplanes, planes, stride=stride, downsample=downsample, groups=groups,
                 base_width=base_width, dilation=dilation, norm_layer=norm_layer)
    self.attention_layer = FPAModule(self.bn2.num_features, 'BatchNorm')

class AdvancedBasicBlock_ATT(AdvancedBasicBlock):
  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, 
               base_width=64, dilation=1, norm_layer=None):
    super(AdvancedBasicBlock_ATT, self).__init__(inplanes, planes, stride=stride, downsample=downsample, groups=groups,
                 base_width=base_width, dilation=dilation, norm_layer=norm_layer)
    self.attention_layer = AttMap(inplanes, planes, stride=stride, downsample=downsample, groups=groups,
                 base_width=base_width, dilation=dilation, norm_layer=norm_layer)

class AdvancedBasicBlock_ATT2(AdvancedBasicBlock):
  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, 
               base_width=64, dilation=1, norm_layer=None):
    super(AdvancedBasicBlock_ATT2, self).__init__(inplanes, planes, stride=stride, downsample=downsample, groups=groups,
                 base_width=base_width, dilation=dilation, norm_layer=norm_layer)
    self.attention_layer = AttMap2(inplanes, planes, stride=stride, downsample=downsample, groups=groups,
                 base_width=base_width, dilation=dilation, norm_layer=norm_layer)

"""## OurResnet"""

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

"""# Adversery Architecture

## Adversery Modules
"""

class BaseNetwork(nn.Module):
  def __init__(self, mode):
    super(BaseNetwork, self).__init__()
    if mode == 'default':
      resnet = newResnet34(BasicBlock, out_channels=10)
    elif mode == 'sam':
      resnet = newResnet34(AdvancedBasicBlock_SAM, out_channels=10)
    elif mode == 'cam':
      resnet = newResnet34(AdvancedBasicBlock_CAM, out_channels=10)
    elif mode == 'fpa':
      resnet = newResnet34(AdvancedBasicBlock_FPA, out_channels=10)
    elif mode == 'att':
      resnet = newResnet34(AdvancedBasicBlock_ATT, out_channels=10)
    elif mode == 'att2':
      resnet = newResnet34(AdvancedBasicBlock_ATT2, out_channels=10)
    elif mode == 'res50':
      resnet = newResnet34(Bottleneck, out_channels=10)
    
    k = load_state_dict_from_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
    #k = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    # k = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
    resnet.load_state_dict(k, strict=False)
    resnet.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    self.model = resnet

  def forward(self, x):
      return self.model(x)

class RaceDetector(nn.Module):
  def __init__(self, in_shape):
    super(RaceDetector, self).__init__()
    self.l1 = nn.Linear(in_shape, 100)
    self.bn1 = nn.BatchNorm1d(100)
    self.d1 = nn.Dropout(p=0.5)
    self.l2 = nn.Linear(100, 1)
  
  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.l1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.d1(x)
    x = self.l2(x)
    x = F.sigmoid(x)
    return x

class Adversary(nn.Module):
  def __init__(self, in_shape):
    super(Adversary, self).__init__()
    self.l1 = nn.Linear(in_shape, 100)
    self.bn1 = nn.BatchNorm1d(100)
    self.l2 = nn.Linear(100, 1)
  
  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.l1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.l2(x)
    return x

"""## Adversery Network Wrapper"""

class AdvNetworkWrapper():
  def __init__(self, basenetwork, racedetector, adversary, bs=50, lr=0.001, rd_epochs = 1, ad_epochs= 1, epochs=10, alpha = 0.8, pretrain_ad_epochs=5, pretrain_rd_epochs=5):
    self.lr = lr
    self.epochs = epochs
    self.batch_size = bs
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.kwargs = {'num_workers': 0, 'pin_memory': True} 
    self.basenetwork = basenetwork.to(self.device)
    self.racedetector = racedetector.to(self.device)
    self.adversary = adversary.to(self.device)
    self.rd_optimizer = optim.Adam(list(self.basenetwork.parameters()) + list(self.racedetector.parameters()), lr=self.lr)
    self.ad_optimizer = optim.Adam(self.adversary.parameters(), lr=self.lr)
    self.rd_criterion = nn.BCELoss()
    self.ad_criterion = nn.SmoothL1Loss()
    self.rd_epochs = rd_epochs
    self.ad_epochs = ad_epochs
    self.pretrain_ad_epochs = pretrain_ad_epochs
    self.pretrain_rd_epochs = pretrain_rd_epochs
    self.epochs = epochs
    self.alpha = alpha
    global global_logger
    self.logger = global_logger
  
  def train(self):
    self.basenetwork.train()
    self.adversary.train()
    self.racedetector.train()

  def eval(self):
    self.basenetwork.eval()
    self.adversary.eval()
    self.racedetector.eval()
  
  def pretrain_rd(self, train_loader, test_loader):
    for epoch in tqdm(range(self.pretrain_rd_epochs)):
      self.train()
      for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(self.device).type(torch.float), target.to(self.device).type(torch.float)
        base_output = self.basenetwork(data)
        rd_output = self.racedetector(base_output)
        self.rd_optimizer.zero_grad()        
        rd_loss = self.rd_criterion(rd_output, target[:,0].unsqueeze(1))
        loss = rd_loss
        loss.backward()
        self.rd_optimizer.step()
        # Bookkeeping
        self.logger.add_scalar('PRE_TRAIN_RDNet_Loss', loss.item(), epoch*len(train_loader) + batch_idx)
        if batch_idx % 20 == 0:
          print('Pre Train Epoch: {} [{}/{} ({:.0f}%)]\tRD Total Loss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        del base_output, rd_output, data, target
        sys.stdout.flush()
      self.test(epoch, test_loader)

  
  def pretrain_ad(self, train_loader):
    for epoch in tqdm(range(self.pretrain_ad_epochs)):
      self.train()
      for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(self.device).type(torch.float), target.to(self.device).type(torch.float)
        base_output = self.basenetwork(data)
        adv_output = self.adversary(base_output.detach())
        self.ad_optimizer.zero_grad()
        loss = self.ad_criterion(adv_output, target[:,1].unsqueeze(1))
        loss.backward()
        self.ad_optimizer.step()

        # Bookkeeping
        self.logger.add_scalar('PRE_TRAIN_AdvNet_AdvLoss', loss.item(), epoch*len(train_loader) + batch_idx)
        if batch_idx % 20 == 0:
          print('Pre Train Epoch: {} [{}/{} ({:.0f}%)]\tAdv Loss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        del base_output, adv_output, data, target

  def train_rd(self, train_loader, network_epoch):
    for epoch in  tqdm(range(self.rd_epochs)):
      self.train()
      for batch_idx, (data, target) in enumerate(train_loader):
        self.train_adv(train_loader, network_epoch*self.rd_epochs*len(train_loader) + epoch*len(train_loader) + batch_idx)
        data, target = data.to(self.device).type(torch.float), target.to(self.device).type(torch.float)
        base_output = self.basenetwork(data)
        rd_output = self.racedetector(base_output)
        adv_output = self.adversary(base_output)
        
        self.rd_optimizer.zero_grad()        
        rd_loss = self.rd_criterion(rd_output, target[:,0].unsqueeze(1))
        adv_loss = self.ad_criterion(adv_output, target[:,1].unsqueeze(1))
        loss = rd_loss - self.alpha * adv_loss
        loss.backward()
        self.rd_optimizer.step()

        # Bookkeeping
        self.logger.add_scalar('RDNet_TotalLoss', loss.item(), network_epoch*self.rd_epochs*len(train_loader) + epoch*len(train_loader) + batch_idx)
        self.logger.add_scalar('RDNet_AdvLoss', adv_loss.item(), network_epoch*self.rd_epochs*len(train_loader) + epoch*len(train_loader) + batch_idx)
        self.logger.add_scalar('RDNet_RDLoss', rd_loss.item(), network_epoch*self.rd_epochs*len(train_loader) + epoch*len(train_loader) + batch_idx)
        if batch_idx % 1 == 0:
          print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tRD Total Loss: {:.6f}'.format(
            epoch, network_epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))       
        del base_output, rd_output, adv_output, data, target
  
  def train_adv(self, train_loader, network_epoch):
    for epoch in tqdm(range(self.ad_epochs)):
      self.train()
      for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(self.device).type(torch.float), target.to(self.device).type(torch.float)
        base_output = self.basenetwork(data)
        adv_output = self.adversary(base_output.detach())
        self.ad_optimizer.zero_grad()
        loss = self.ad_criterion(adv_output, target[:,1].unsqueeze(1))
        loss.backward()
        self.ad_optimizer.step()

        # Bookkeeping
        self.logger.add_scalar('AdvNet_AdvLoss', loss.item(), network_epoch*self.ad_epochs*len(train_loader) + epoch*len(train_loader) + batch_idx)
        if batch_idx % 20 == 0:
          print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tAdv Loss: {:.6f}'.format(
            epoch, network_epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        del base_output, adv_output, data, target
        
  def train_network(self, train_loader, test_loader):
    #self.pretrain_rd(train_loader, test_loader)
    self.test(0, test_loader)
    self.test(1, test_loader)
    self.test(2, test_loader)
    self.test(3, test_loader)
    self.test(4, test_loader)
    self.test(5, test_loader)
    self.test(6, test_loader)
    self.test(7, test_loader)
    self.test(8, test_loader)
    self.test(9, test_loader)
    self.test(10, test_loader)
    #self.pretrain_ad(train_loader)
    #for epoch in tqdm(range(self.epochs)):
      # self.train_adv(train_loader, epoch)
      #self.train_rd(train_loader, epoch)
      #self.test(epoch, test_loader)  

  # def calculate_bias(pred_list, true_list, bmi_list):
  #   d = {'BMI': bmi_list, 'truthLabel': true_list, 'predLabel': pred_list}
  #   predDf = pd.DataFrame(d, columns=['BMI','truthLabel','predLabel'])
  #   outputData = []
  #   for i in range(8):
  #     bmi_min,bmi_max = 15 + 5*i, 15 + 5*(i+1)
  #     subDf = predDf[predDf.BMI.between(bmi_min,bmi_max)]
     
  #     whiteDf = subDf[subDf.truthLabel==1]
  #     aaDf = subDf[subDf.truthLabel==0]
         
  #     tempDf = pd.concat([whiteDf, aaDf])
     
  #     whiteAcc = np.round(accuracy_score(whiteDf.truthLabel, whiteDf.predLabel),2)
  #     aaAcc = np.round(accuracy_score(aaDf.truthLabel, aaDf.predLabel),2)
     
  #     acc = np.round(accuracy_score(tempDf.truthLabel,tempDf.predLabel),2)
     
  #     outputData.append([bmi_min, bmi_max, len(subDf), acc, whiteAcc,aaAcc])

       
  #   tempDf = pd.DataFrame(outputData)
  #   tempDf.columns = ['BMI_min','BMI_max','numImages','Total Acc','White Acc','Black Acc']
  #   totalDiff = 0
  #   for i in range(len(tempDf)):
  #     diff = np.abs(tempDf.iloc[i]['White Acc'] - tempDf.iloc[i]['Black Acc'])     
  #     totalDiff += diff
  #   biasScore = totalDiff/len(tempDf)
  #   return biasScore 


  def test(self, epoch, test_loader):
    self.eval()
    test_loss = 0
    test_adv_loss = 0
    test_rd_loss = 0
    correct = 0
    global global_epoch, global_batchid, global_logger
    global_epoch = epoch 
    with torch.no_grad():
        for batch_id, tup in enumerate(test_loader):
            global_batchid = batch_id
            data, target = tup
            data, target = data.to(self.device).type(torch.float), target.to(self.device).type(torch.float)
            # plt.imshow(data[0].transpose(0,1).transpose(1,2).squeeze())
            # plt.show()
            base_output = self.basenetwork(data)
            rd_output = self.racedetector(base_output)
            adv_output = self.adversary(base_output)

            test_rd_loss += self.rd_criterion(rd_output, target[:,0].unsqueeze(1)).item()
            test_adv_loss += self.ad_criterion(adv_output, target[:,1].unsqueeze(1)).item()
            test_loss = test_rd_loss + test_adv_loss
            
            pred = (rd_output > 0.5).type(torch.float)
            correct += pred.eq(target[:,0].view_as(pred)).sum().item()
            if(batch_id == 0):
              print(data.shape)
              global_logger.add_images("Input Image 1", vutils.make_grid(data.cpu()[:50, :, :, :], padding=2).unsqueeze(0),  epoch)
              global_logger.add_images("Input Image 2", vutils.make_grid(data.cpu()[50:, :, :, :], padding=2).unsqueeze(0),  epoch)
              print("Logged")
            del data, base_output, rd_output, adv_output

    test_loss /= len(test_loader)
    test_adv_loss /= len(test_loader)
    test_rd_loss /= len(test_loader)

    self.logger.add_scalar('Test_Loss', test_loss, epoch)
    self.logger.add_scalar('Test_Adv_Loss', test_adv_loss, epoch)
    self.logger.add_scalar('Test_Rd_Loss', test_rd_loss, epoch)
    self.logger.add_scalar('Test_Accuracy', correct/len(test_loader.dataset), epoch)

    print('\nTest set: Loss: {:.4f}, RD Loss: {:.4f}, AD Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_rd_loss, test_adv_loss , correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    sys.stdout.flush()

"""# Build Model"""

# df = pd.read_csv('/cbica/home/thodupuv/acv/FairRaceDetection/LabelFile_2000.csv')
# df['BMI'] = (df['BMI'] - df['BMI'].min())/(df['BMI'].max() - df['BMI'].min())
# df.head()
# trainData = df[df.train == False]
# validData = df[df.train == True]

# dataFolder = "/cbica/home/santhosr/CBIG/BIRAD/Data"
# train_dataset = CancerDatasetNP(dataFolder, trainData) 
# test_dataset = CancerDatasetNP(dataFolder, validData) 

train_dataset = CancerDatasetNP('/cbica/home/thodupuv/acv/data/Cancer/train_full_256.pk') 
test_dataset = CancerDatasetNP('/cbica/home/thodupuv/acv/data/Cancer/val_full_256.pk') 
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)


bn = BaseNetwork('att2')
rn = RaceDetector(512)
an = Adversary(512)

adnw = AdvNetworkWrapper(bn, rn, an, bs=50, lr=0.00001, rd_epochs = 1, ad_epochs= 1, epochs=10, pretrain_rd_epochs=200, alpha = 5)

##################################################
############## Custom LR #########################
##################################################
lr = 1e-5
lr_cust = 1e-6
plist =  []
for name, param in list(adnw.basenetwork.named_parameters()):
  pa = {}
  pa["params"] = param
  pa["lr"] = lr
  if not ('attention' in name or 'model.conv1' in name):
    pa["lr"] = lr_cust
  plist.append(pa)

for name, param in list(adnw.racedetector.named_parameters()):
  pa = {}
  pa["params"] = param
  pa["lr"] = lr
  plist.append(pa)
adnw.rd_optimizer = torch.optim.Adam(plist, lr=lr)
##################################################
##################################################
adnw.train_network(train_loader, test_loader)
torch.save(adnw.basenetwork, exp_folder + '_basenetwork.pt')
torch.save(adnw.racedetector, exp_folder + '_racedetector.pt')
torch.save(adnw.adversary, exp_folder + '_adversary.pt')

