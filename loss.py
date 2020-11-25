import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable



def predictionLoss(y,y_hat,N=48):
    loss = (((y-y_hat)**2)/N).sum()
    loss = loss.cpu()
    loss = np.sqrt(loss.detach().numpy())
    loss = torch.tensor(loss).cuda()
    loss = Variable(loss,requires_grad=True)
    return loss

def rmse(y,y_hat,N):
    loss = np.sqrt(((y-y_hat)**2).sum().item()/N)
    return loss


def nrmse(y,y_hat,N):
    RMSE = rmse(y,y_hat,N)
    loss = RMSE/(y.max().item()-y_hat.min().item())
    return loss

def mae(y,y_hat,N):
    loss = (abs(y-y_hat).sum().item())/N
    return loss

def getloss(y,y_hat):
    loss = ((y-y_hat)**2).sum(axis=3)
    loss = np.sqrt(loss).view([-1,1])
    return loss



if __name__ == "__main__":
    y = torch.randn([64,1,1,48]).cuda()
    y_hat = torch.randn([64,48]).cuda()
    loss = predictionLoss(y,y_hat)
    print(loss)
    print(loss.shape)
    print(type(loss))