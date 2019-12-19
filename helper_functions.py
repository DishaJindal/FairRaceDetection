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
import subprocess

def printNvidiaSmi():
  sp = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  out_str = sp.communicate()
  out_list = out_str[0].decode("utf-8").split('\n')

  for i in out_list:
    print(i)

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

def get_tensorboard_logger():
  logs_base_dir = "/cbica/home/thodupuv/acv/logs"
  logs_dir = logs_base_dir + "/run_" + "_" + str(time.mktime(datetime.datetime.now().timetuple()))
  os.makedirs(logs_dir, exist_ok=True)
  from torch.utils.tensorboard import SummaryWriter
  # %tensorboard --logdir {logs_base_dir} --port=9002
  logger = SummaryWriter(logs_dir)
  return logger