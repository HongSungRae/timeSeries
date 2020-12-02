import numpy as np
import torch
import torch.nn as nn
#from torch.autograd import Variable



def predictionLoss(y,y_hat):
    y_hat = y_hat.view(y_hat.shape[0],1,1,-1)
    loss = torch.sqrt(torch.mean((y-y_hat)**2))
    return loss

'''
def predictionLoss(y,y_hat,N=48): #이거 그레디언트 계속 아마 0일거다
    loss = (((y-y_hat)**2)/N).sum()
    loss = loss.cpu()
    loss = np.sqrt(loss.detach().numpy())
    loss = torch.tensor(loss).cuda()
    loss = Variable(loss,requires_grad=True)
    return loss
'''
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
    y = torch.randn([64,1,1,48])
    y_hat = torch.randn([64,48])
    loss = predictionLoss(y,y_hat)
    print(loss)
    print(loss.shape)
    print(loss.requires_grad)

    criteria = nn.MSELoss()
    loss = criteria(y,y_hat.view(y_hat.shape[0],1,1,48))
    print(loss)
    print(loss.shape)
    print(loss.requires_grad)