import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pandas as pd
import numpy as np



class LoadCNN(nn.Module):
  def __init__(self, in_channels = 1, factor): # factor = ID + Month + Day + Week
    super().__init__()

    self.factor = factor
    self.CNN_channels = CNN_channels()
    self.linear = nn.Linear(in_features=1024,out_features=48)

  def forward(self,x):
    x = self.CNN_channels(x)
    x = torch.cat(x,self,factor)
    x = x.reshape(x.shape[0],-1) # flatten
    x = self.linear(x)
    return x


class CNN_channels(nn.Module):
  def __init__(self, in_channesl = 1):
    super().__init__()

    self.horizontal_channel = horizontal()
    self.vertical_channel = vertical()
    self.dropout = nn.Dropout(p=.5)

  def forward(self,x):
    return dropout(torch.cat([self.horizontal(x), self.vertical(x)]))


class horizontal(nn.Module):
  def __init__(self, in_channels = 1):
    super().__init__()

    self.conv1 = conv_block(in_channels=in_channels, out_channels = 16, kernel_size=(1,7))
    self.conv2 = conv_block(16, 24, kernel_size=(1,5))
    self.maxPool1 = nn.MaxPool2d(kernel_size=(1,2))
    self.conv3 = conv_block(24, 24, kernel_size =(1,5))
    self.maxPool2 = nn.MaxPool2d(kernel_size=(1,2))
    self.conv4 = conv_block(24, 64, kernel_size =(1,4))
    self.maxPool3 = nn.MaxPool2d(kernel_size=(2,1))
    self.conv5 = conv_block(64, 64, kernel_size =(1,3))
    self.maxPool4 = nn.MaxPool2d(kernel_size=(2,1))
    self.conv6 = conv_block(64, 64, kernel_size =(1,3))

  def forward(self,x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.maxPool1(x)
    x = self.conv3(x)
    x = self.maxPool2(x)
    x = self.conv4(x)
    x = self.maxPool3(x)
    x = self.conv5(x)
    x = self.maxPool4(x)
    x = self.conv6(x)
    return x


class vertical(nn.Module):
  def __init__(self, in_channels = 1):
    super().__init__()

    self.conv1 = conv_block(in_channels=in_channels, out_channels=16, kernel_size=(4,1))
    self.maxPool1 = nn.MaxPool2d(kernel_size=(1,2))
    self.conv2 = conv_block(16, 24, kernel_size=(4,1))
    self.maxPool2 = nn.MaxPool2d(kernel_size=(1,2))
    self.conv3 = conv_block(24, 24, kernel_size=(3,1))
    self.conv4 = conv_block(24, 64, kernel_size=(3,1))
    self.maxPool3 = nn.MaxPool2d(kernel_size=(1,2))
    self.conv5 = conv_block(64, 64, kernel_size=(2,1))
    self.maxPool4 = nn.MaxPool2d(kernel_size=(2,1))
    self.conv6 = conv_block(64, 64, kernel_size=(2,1))

  def forward(self,x):
    x = self.conv1(x)
    x = self.maxPool1(x)
    x = self.conv2(x)
    x = self.maxPool2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.maxPool3(x)
    x = self.conv5(x)
    x = self.maxPool4(x)
    x = self.conv6(x)
    return x


class conv_block(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs): # nn.conv2d의 나머지 파라메타도 그대로 가져다 쓰겠다
    super().__init__()
    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(in_channels,out_channels,**kwargs)

  def forward(self,x):
    return self.relu(self.conv(x))


if __name__ == "__main__":
    net = CNN_channels().cuda()
    summary(net,input_size=(1,7,48))