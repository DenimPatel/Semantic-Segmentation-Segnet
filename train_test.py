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

# models
from models import segnet_modified_with_skip as segnet_s
from models import segnet_modified as segnet_mo
from models import segnet as segnet_b
from pretrained_weights import load_pretrained_weights

# provide path to dataset
DATA_PATH = 'DATA PATH'

# Using Cityscapes dataset
train = datasets.Cityscapes(DATA_PATH, split = 'train', mode = 'fine', target_type = 'semantic',transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]),target_transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]))
test = datasets.Cityscapes(DATA_PATH, split = 'test', mode = 'fine', target_type = 'semantic' ,transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]),target_transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]))
val = datasets.Cityscapes(DATA_PATH, split = 'val', mode = 'fine', target_type = 'semantic' ,transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]),target_transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]))

# get train-test-validation dataloader ; bathch size is small due to GPU computation constaint
trainset = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=2, shuffle=False)
valset = torch.utils.data.DataLoader(val, batch_size=2, shuffle=True)

def main():
  # select GPU if available 
  if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Running on the GPU")
  else:
    device = torch.device("cpu")
    print("Running on the CPU")
    
    """ Use any of the following model for segmentation
     segnet_b.SegNet_sequential().to(device)  Segnet architecture
     segnet_mo.SegNet_sequential().to(device) modified SegNet architecture
     segnet_s.SegNet_sequential().to(device)  modified SegNet with Skip connections
    """

  net = segnet_b.SegNet_sequential().to(device)              
  print(net)                                                  
  return net,device                                           
       
def train(net, device):
  """
  trains the Segmentation model
  - reports loss per opoch while training
  - saves the trained network 
  """
  epochs = 100
  weight = torch.ones(34)

  # Loss function is pixel level classification error
  loss_function = nn.CrossEntropyLoss(weight).cuda()
  
  # Adam optimizer 
  optimizer = optim.Adam(net.parameters(), lr=0.01)
  
  for epoch in range(epochs): 
    for data in trainset:

        # get the batch of data
        X, y = data

        # send to the Device i.e. GPU if present
        X, y = X.to(device), y.to(device)

        # forwad pass
        output = net(X)

        output = output.view(output.size(0),output.size(1), -1)     
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))

        # scale to 255 becuse prection score will be between 0 to 1
        label = y*255
        label = label.long().view(-1)

        # calculate loss on the prediction
        loss = loss_function(output, label)

        # backward pass
        loss.backward()  
        optimizer.step()

        # reset gradients
        net.zero_grad()  

    print("Epoch No:",epoch)
    print(loss) 
    torch.save(net.state_dict(),'wts_segnet.pth')  #For saving weights after every Epoch 

def decode_segmap(image, classes = 31):
  """
  Colorcodes the result of segmentation result for visualization
  """
  # color coding scheme 
  label_colors = np.array([(0, 0, 0),  
               (128, 0, 0), (128,64,128), (128, 128, 0), (0, 0, 50), (128, 0, 128),
               (0, 128, 64), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               (198, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
               (0,135,0),(128,192,128),(64,128,192),(220,20,60),(64,192,128),               
               (0, 0,190),(128,128,192),(128,192,64),(128,64,192),(192,64,128)])
 
  # create three channel of same size as image
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)

  # assigne pixel values based on class 
  for l in range(0, classes):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
     
  rgb = np.stack([r, g, b], axis=2)
  return rgb

def test(net, device):
  """
  Tests the accuracy of the segmentation result
  Reports accuracy based on pixel level incorrect labels 
  """
  correct = 0
  valset_size = 500
  print("Testing")
  with torch.no_grad(): #as we are not training the model
      nonzerocount = 0 #calculates total wrong classified pixels in the validation data
      for data in valset:
          
          # get test batch
          X2, y = data
          X, y = X2.to(device), y.to(device)
          
          # forward pass
          output = net(X)

          # calculate accuracy of the model
          for idx in range(len(output)):
            out = output[idx].cpu().max(0)[1].data.squeeze(0).byte().numpy()
            predicted_mx = output[idx]
            predicted_mx_idx = torch.argmax(predicted_mx,0)
            predicted_mx_idx = predicted_mx_idx.detach().cpu().numpy()           
            rgb = decode_segmap(predicted_mx_idx)
            fig = plt.figure(1)
            plt.imshow(rgb)
            plt.figure(2)
            plt.imshow(transforms.ToPILImage()(data[0][idx]))#.detach().cpu().numpy())
            plt.show()
            label = y[idx][0].detach().cpu().numpy()
            final_diff = predicted_mx_idx - label*255
            nonzerocount = nonzerocount + np.count_nonzero(final_diff)
      accu = 1 - nonzerocount/(valset_size*256*512)
      print("Accuracy",accu)

if __name__ = '__main__':
  if (len(sys.argv) != 2):
    print("== INCORRECT INPUT ==")
    print("USE: python train_test.py train to train the model")
    print("USE: python train_test.py test to test the model")
    exit()

  sec = time.time()
  net, device = main()
  net = load_pretrained_weights(net)

  if (sys.argv[1] == "train"):
    train(net, device)                               
  elif (sys.argv[1] == "test"):
    net.load_state_dict(torch.load('wts_segnet.pth'))          
    test(net, device)                             
  sec_last = time.time()
  print("Execution took ",f'{sec_last-sec:.2f}'," seconds")