import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from loadLayer import LoadCNN
from dataloader import myDataLoader
from loss import *



def train(model,train_loader,test_loader,epoch):
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    net = model
    optimizer = optim.SGD(model.parameters(),lr=1e-3)
    # criterion = nn.??
    epochs = epoch
    total_batch = len(train_loader)
    print('total batch :',total_batch)

    train_loss_list = []
    test_loss_list = []

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
            y_hat = net(x,factor)
            loss = predictionLoss(y_hat, target)
            loss.backward()
            optimizer.step()

            loss_learning += loss

            #del loss
            #del y_hat
           
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('[epoch : %d, iter : %5d] loss: %.3f' %
                      (eph + 1, i + 1, loss_learning / (i+1)))

            if i == total_batch-1: # 마지막 미니배치
                with torch.no_grad():
                    x, target, factor = next(iter(test_loader))
                    if is_cuda:
                        x = x.float().cuda()
                        target = target.float().cuda()
                        factor = factor.float().cuda()
                    y_hat = net(x,factor)
                    loss = predictionLoss(y_hat,target)
                    test_loss_list.append(loss)

                    train_loss_list.append(loss_learning/total_batch)
                    loss_learning = 0.0


    print('Finished Training')
    return net, train_loss_list, test_loss_list




if __name__ == "__main__":
    df_train = pd.read_csv("/daintlab/data/sr/traindf.csv",index_col=0)

    train_dataset = myDataLoader(df_train)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=64, pin_memory=False)

    df_test = pd.read_csv("/daintlab/data/sr/testdf.csv",index_col=0)

    test_dataset = myDataLoader(df_test)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=64, pin_memory=False)

    model = LoadCNN().cuda()
    trained_model, train_loss_list, test_loss_list = train(model,train_loader,test_loader,20)

    print(train_loss_list)
    print(test_loss_list)

    PATH = '/daintlab/data/sr/'
    torch.save(trained_model, PATH + 'LoadCNNmodel.pt')