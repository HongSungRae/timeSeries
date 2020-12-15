import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import pandas as pd
import numpy as np

from loadLayer import LoadCNN
from dataloader import myDataLoader
from loss import *


def kfold(train_dataset,k):
    total_size = len(train_dataset)
    val_size = int(total_size/k)
    train_size = total_size - val_size

    train_dataset, validation_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, sampler = sampler_(train_dataset), pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=64, sampler = sampler_(val_dataset), pin_memory=True)



def trainKfld(model,train_loader,epoch):
    is_cuda = torch.cuda.is_available() # True or False
    device = torch.device('cuda' if is_cuda else 'cpu') 

    model = model
    optimizer = optim.SGD(model.parameters(),lr=1e-3)
    #criterion = nn.MSELoss()
    #eps = 1e-6
    epochs = epoch
    total_batch = len(train_loader)
    print('total batch :',total_batch)

    train_loss_list = []
    val_loss_list = []

    for eph in range(epochs):
        print('epoch / epochs = {} / {}'.format(eph+1,epochs))
        loss_learning = 0.0
        for i,data in enumerate(train_loader):
            x, target, factor = data
            if is_cuda:
                x = x.float().cuda()
                target = target.float().cuda()
                factor = factor.float().cuda()
            
            optimizer.zero_grad()
            y_hat = model(x,factor)
            #y_hat = y_hat.view(y_hat.shape[0],1,1,-1)
            #loss = torch.sqrt(criterion(target, y_hat) + eps)
            loss = predictionLoss(y_hat, target)
            loss.backward()
            optimizer.step()

            loss_learning += loss

            #del loss
            #del y_hat
            #print(torch.sum(net.linear.weight))
           
            if i % 100 == 99:    # print every 1000 mini-batches
                print('[epoch : %d, iter : %5d] loss: %.3f' %
                      (eph + 1, i + 1, loss_learning / (i+1)))
                print(torch.sum(net.linear.weight))

            if i == 999: # total_batch-1
                train_loss_list.append(loss_learning.item()/1000)#total_batch
                loss_learning = 0.0
                
                with torch.no_grad():
                    loss_test = 0.0
                    for j,data in enumerate(test_loader):
                        x, target, factor = data
                        if is_cuda:
                            x = x.float().cuda()
                            target = target.float().cuda()
                            factor = factor.float().cuda()
                        y_hat = net(x,factor)
                        loss = predictionLoss(y_hat,target)
                        loss_test += loss
                        
                        if j==9:
                            test_loss_list.append(loss_test.item()/10)
                            break
                break
            

    print('Finished Training')
    return model, train_loss_list, test_loss_list




if __name__ == "__main__":
    df_train = pd.read_csv("/daintlab/data/sr/traindf.csv",index_col=0)

    train_dataset = myDataLoader(df_train)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64, pin_memory=True)

    df_test = pd.read_csv("/daintlab/data/sr/testdf.csv",index_col=0)

    test_dataset = myDataLoader(df_test)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=64, pin_memory=True)

    model = LoadCNN().cuda()
    trained_model, train_loss_list, test_loss_list = train(model,train_loader,test_loader,10)

    print(train_loss_list)
    print(test_loss_list)

    PATH = '/daintlab/data/sr/'
    torch.save(trained_model, PATH + 'LoadCNNmodel.pt')