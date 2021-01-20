import torch.nn.functional as F
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
import os
import sys
from dataloader import MyDataLoader
from torch.utils.data import Dataset, DataLoader
from model import Model
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from loss import RMSE


def train(model,train_loader,test_loader,epoch):
    is_cuda = torch.cuda.is_available() # True or False
    device = torch.device('cuda' if is_cuda else 'cpu') 

    net = model
    optimizer = optim.SGD(net.parameters(),lr=1e-2)
    criterion = RMSE()
    #eps = 1e-6
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
            #y_hat = y_hat.view(y_hat.shape[0],1,1,-1)
            #loss = torch.sqrt(criterion(target, y_hat) + eps)
            loss = criterion(target,y_hat)
            #loss = predictionLoss(y_hat, target)
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
                        #loss = predictionLoss(y_hat,target)
                        loss_ = criterion(target,y_hat)
                        loss_test += loss_
                        
                        if j==99:
                            test_loss_list.append(loss_test.item()/100)
                            print("test loss : {0}".format(loss_test.item()/100))
                            break
                break
            

    print('Finished Training')
    return net, train_loss_list, test_loss_list



if __name__=='__main__':
    df_train = pd.read_csv("/daintlab/data/sr/traindf.csv",index_col=0)

    train_dataset = MyDataLoader(df_train)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128, pin_memory=True)

    df_test = pd.read_csv("/daintlab/data/sr/testdf.csv",index_col=0)

    test_dataset = MyDataLoader(df_test)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=128, pin_memory=True)

    lstm = Model(
        input_size=48*7,
        hidden_size=256,
        num_layers=3
    ).cuda()
    trained_model, train_loss_list, test_loss_list = train(lstm,train_loader,test_loader,10)

    print(train_loss_list)
    print(test_loss_list)

    PATH = '/daintlab/data/sr/'
    torch.save(trained_model, PATH + 'LSTMmodel.pt')