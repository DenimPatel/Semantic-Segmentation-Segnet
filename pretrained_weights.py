import numpy as np 
import time
from PIL import Image 
import matplotlib.pyplot as plt
import sys

# pytorch related libraries/methods
import torch
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as standard_transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def load_pretrained_weights(net):
  """
  uses transfer learning approaoch to reduce the training time 
  VGG16 weights are used
  """
  vgg = torchvision.models.vgg16(pretrained=True,progress=True)
  pretrained_dict = vgg.state_dict()
  model_dict = net.state_dict()
  list1 = ['layer10_conv.weight',
    'layer10_conv.bias',
    'layer11_conv.weight',
    'layer11_conv.bias',
    'layer20_conv.weight',
    'layer20_conv.bias',
    'layer21_conv.weight',
    'layer21_conv.bias',
    'layer30_conv.weight',
    'layer30_conv.bias',
    'layer31_conv.weight',
    'layer31_conv.bias',
    'layer32_conv.weight',
    'layer32_conv.bias',
    'layer40_conv.weight',
    'layer40_conv.bias',
    'layer41_conv.weight',
    'layer41_conv.bias',
    'layer42_conv.weight',
    'layer42_conv.bias',
    'layer50_conv.weight',
    'layer50_conv.bias',
    'layer51_conv.weight',
    'layer51_conv.bias',
    'layer52_conv.weight',
    'layer52_conv.bias'
    ]
  list2 = ['features.0.weight',
    'features.0.bias',
    'features.2.weight',
    'features.2.bias',
    'features.5.weight',
    'features.5.bias',
    'features.7.weight',
    'features.7.bias',
    'features.10.weight',
    'features.10.bias',
    'features.12.weight',
    'features.12.bias',
    'features.14.weight',
    'features.14.bias',
    'features.17.weight',
    'features.17.bias',
    'features.19.weight',
    'features.19.bias',
    'features.21.weight',
    'features.21.bias',
    'features.24.weight',
    'features.24.bias',
    'features.26.weight',
    'features.26.bias',
    'features.28.weight',
    'features.28.bias'
    ]
  for l in range(len(list1)):
    pretrained_dict[list1[l]] = pretrained_dict.pop(list2[l])

  pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict)
  net.load_state_dict(model_dict)
  return net