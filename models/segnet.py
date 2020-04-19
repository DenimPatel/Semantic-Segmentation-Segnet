import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

class SegNet_sequential(nn.Module) :
  """
  Implementation of Vanilla Segnet
  """
  def __init__(self):
    super(SegNet_sequential,self).__init__()
        
    self.layer10_conv = nn.Conv2d(3,64,3,1,padding = (1,1))
    self.layer10_batch = nn.BatchNorm2d(64, affine = False)
    
    self.layer11_conv = nn.Conv2d(64,64,3,1,padding=(1,1))
    self.layer11_batch = nn.BatchNorm2d(64, affine = False)
    
    self.layer20_conv = nn.Conv2d(64,128,3,1,padding=(1,1))
    self.layer20_batch = nn.BatchNorm2d(128, affine = False)
    
    self.layer21_conv = nn.Conv2d(128,128,3,1,padding=(1,1))
    self.layer21_batch = nn.BatchNorm2d(128, affine = False)
    
    self.layer30_conv = nn.Conv2d(128,256,3,1,padding=(1,1))
    self.layer30_batch = nn.BatchNorm2d(256, affine = False)
    
    self.layer31_conv = nn.Conv2d(256,256,3,1,padding=(1,1))
    self.layer31_batch = nn.BatchNorm2d(256, affine = False)
    
    self.layer32_conv = nn.Conv2d(256,256,3,1,padding=(1,1))
    self.layer32_batch =  nn.BatchNorm2d(256, affine = False)
    
    self.layer40_conv = nn.Conv2d(256,512,3,1,padding=(1,1))
    self.layer40_batch = nn.BatchNorm2d(512, affine = False)
    
    self.layer41_conv = nn.Conv2d(512,512,3,1,padding=(1,1))
    self.layer41_batch = nn.BatchNorm2d(512, affine = False)
    
    self.layer42_conv = nn.Conv2d(512,512,3,1,padding=(1,1))
    self.layer42_batch = nn.BatchNorm2d(512, affine = False)
    
    self.layer50_conv = nn.Conv2d(512,512,3,1,padding=(1,1))
    self.layer50_batch = nn.BatchNorm2d(512, affine = False)
    
    self.layer51_conv = nn.Conv2d(512,512,3,1,padding=(1,1))
    self.layer51_batch = nn.BatchNorm2d(512, affine = False)
    
    self.layer52_conv = nn.Conv2d(512,512,3,1,padding=(1,1))
    self.layer52_batch = nn.BatchNorm2d(512, affine = False)
    

    self.decoder_layer52_conv = nn.ConvTranspose2d(512,512,3,1,padding=(1,1))
    self.decoder_layer52_batch = nn.BatchNorm2d(512, affine = False)
    
    self.decoder_layer51_conv = nn.ConvTranspose2d(512,512,3,1,padding=(1,1))
    self.decoder_layer51_batch = nn.BatchNorm2d(512, affine = False)
    
    self.decoder_layer50_conv = nn.ConvTranspose2d(512,512,3,1,padding=(1,1))
    self.decoder_layer50_batch = nn.BatchNorm2d(512, affine = False)

    self.decoder_layer42_conv = nn.ConvTranspose2d(512,512,3,1,padding=(1,1))
    self.decoder_layer42_batch = nn.BatchNorm2d(512, affine = False)
    
    self.decoder_layer41_conv = nn.ConvTranspose2d(512,512,3,1,padding=(1,1))
    self.decoder_layer41_batch = nn.BatchNorm2d(512, affine = False)
    
    self.decoder_layer40_conv = nn.ConvTranspose2d(512,256,3,1,padding=(1,1))
    self.decoder_layer40_batch = nn.BatchNorm2d(256, affine = False)
    
    self.decoder_layer32_conv = nn.ConvTranspose2d(256,256,3,1,padding=(1,1))
    self.decoder_layer32_batch = nn.BatchNorm2d(256, affine = False)
    
    self.decoder_layer31_conv = nn.ConvTranspose2d(256,256,3,1,padding=(1,1))
    self.decoder_layer31_batch = nn.BatchNorm2d(256, affine = False)
    
    self.decoder_layer30_conv = nn.ConvTranspose2d(256,128,3,1,padding=(1,1))
    self.decoder_layer30_batch = nn.BatchNorm2d(128, affine = False)
    
    self.decoder_layer21_conv = nn.ConvTranspose2d(128,128,3,1,padding=(1,1))
    self.decoder_layer21_batch = nn.BatchNorm2d(128, affine = False)
    
    self.decoder_layer20_conv = nn.ConvTranspose2d(128,64,3,1,padding=(1,1))
    self.decoder_layer20_batch = nn.BatchNorm2d(64, affine = False)
    
    self.decoder_layer11_conv = nn.ConvTranspose2d(64,64,3,1,padding=(1,1))
    self.decoder_layer11_batch = nn.BatchNorm2d(64, affine = False)
    
    self.decoder_layer10_conv = nn.ConvTranspose2d(64,34,3,1,padding=(1,1))
    
  def forward(self,x):

    x10 = F.relu(self.layer10_batch(self.layer10_conv(x)))
    x11 = F.relu(self.layer11_batch(self.layer11_conv(x10)))
    x1 , x1_indices = F.max_pool2d(x11,kernel_size=2,stride=2,return_indices=True)

    x20 = F.relu(self.layer20_batch(self.layer20_conv(x1)))
    x21 = F.relu(self.layer21_batch(self.layer21_conv(x20)))
    x2, x2_indices = F.max_pool2d(x21,kernel_size=2,stride=2,return_indices=True)

    x30 = F.relu(self.layer30_batch(self.layer30_conv(x2)))
    x31 = F.relu(self.layer31_batch(self.layer31_conv(x30)))
    x32 = F.relu(self.layer32_batch(self.layer32_conv(x31)))
    x3 , x3_indices = F.max_pool2d(x32,kernel_size=2,stride=2,return_indices=True)

    x40 = F.relu(self.layer40_batch(self.layer40_conv(x3)))
    x41 = F.relu(self.layer41_batch(self.layer41_conv(x40)))
    x42 = F.relu(self.layer42_batch(self.layer42_conv(x41)))
    x4, x4_indices = F.max_pool2d(x42,kernel_size=2,stride=2,return_indices=True)

    x50 = F.relu(self.layer50_batch(self.layer50_conv(x4)))
    x51 = F.relu(self.layer51_batch(self.layer51_conv(x50)))
    x52 = F.relu(self.layer52_batch(self.layer52_conv(x51)))
    x5, x5_indices = F.max_pool2d(x52,kernel_size=2,stride=2,return_indices=True)

    x5_decoder_output = F.max_unpool2d(x5,x5_indices,kernel_size = 2,stride = 2)
    x52_dec = F.relu(self.decoder_layer52_batch(self.decoder_layer52_conv(x5_decoder_output)))
    x51_dec = F.relu(self.decoder_layer51_batch(self.decoder_layer51_conv(x52_dec)))
    x50_dec = F.relu(self.decoder_layer50_batch(self.decoder_layer50_conv(x51_dec)))

    x4_decoder_output = F.max_unpool2d(x50_dec,x4_indices,kernel_size=2,stride=2)
    x42_dec = F.relu(self.decoder_layer42_batch(self.decoder_layer42_conv(x4_decoder_output)))
    x41_dec = F.relu(self.decoder_layer41_batch(self.decoder_layer41_conv(x42_dec)))
    x40_dec = F.relu(self.decoder_layer40_batch(self.decoder_layer40_conv(x41_dec)))

    x3_decoder_output = F.max_unpool2d(x40_dec,x3_indices,kernel_size = 2,stride=2)
    x32_dec = F.relu(self.decoder_layer32_batch(self.decoder_layer32_conv(x3_decoder_output)))
    x31_dec = F.relu(self.decoder_layer31_batch(self.decoder_layer31_conv(x32_dec)))
    x30_dec = F.relu(self.decoder_layer30_batch(self.decoder_layer30_conv(x31_dec)))

    x2_decoder_output = F.max_unpool2d(x30_dec,x2_indices,kernel_size = 2,stride=2)
    x21_dec = F.relu(self.decoder_layer21_batch(self.decoder_layer21_conv(x2_decoder_output)))
    x20_dec = F.relu(self.decoder_layer20_batch(self.decoder_layer20_conv(x21_dec)))

    x1_decoder_output = F.max_unpool2d(x20_dec,x1_indices,kernel_size= 2,stride=2)
    x11_dec = F.relu(self.decoder_layer11_batch(self.decoder_layer11_conv(x1_decoder_output)))
    x10_dec = F.relu(self.decoder_layer10_conv(x11_dec))
    
    x_out = F.softmax(x10_dec,dim=1)

    return x_out