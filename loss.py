import numpy as np
import torch


def predictionLoss(y,y_hat,N=48):
    loss = ((y-y_hat)**2).sum(axis=3)/N
    loss = loss.cpu()
    return np.sqrt(loss).view([-1,1,1,1]).cuda()


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