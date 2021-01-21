import torch.nn.functional as F
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
from dataloader import MyDataLoader
from torch.utils.data import Dataset, DataLoader
from model import Model
from loss import RMSEforLSTM
#import os
#import sys
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
#from loss import RMSE


def train(model,train_loader,test_loader,epoch):
    is_cuda = torch.cuda.is_available() # True or False
    device = torch.device('cuda' if is_cuda else 'cpu') 

    net = model
    optimizer = optim.SGD(net.parameters(),lr=1e-2)
    criterion = RMSEforLSTM()
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
            loss = criterion(target,y_hat)
            loss_learning += loss # 여기에 loss 더하는 순간 loss_learining은 tensor가 된다
                                  # loss를 total_loss에 더해서 나중에 이터레이션 수만큼 나누는 행위때문에
                                  # loss가 뒤로 갈수록 잘 안떨어지고 (분자가 분모에비해 커져서)
                                  # 새 epoch이 시작될때마다 loss가 크게 떨어지는 것이였다! (분모가 상대적으로 커짐)
            loss.backward()
            optimizer.step()
           
            if i % 100 == 99:    # print every 100 mini-batches
                print('[epoch : %d, iter : %5d] loss: %.3f, sum(loss): %.3f' %
                      (eph+1, i+1, loss_learning.item()/100,loss_learning.item()))
                loss_learning = 0.0 # total_loss 위의 문제 해결 위해 여기서 초기화해줌
                #print(torch.sum(net.linear2.weight))

            if i == total_batch-1:
                train_loss_list.append(loss_learning.item()/(total_batch%100))
                
                with torch.no_grad():
                    loss_test = 0.0
                    for j,data in enumerate(test_loader):
                        x, target, factor = data
                        if is_cuda:
                            x = x.float().cuda()
                            target = target.float().cuda()
                            factor = factor.float().cuda()
                        
                        y_hat = net(x,factor)
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
        hidden_size=48,
        num_layers=3
    ).cuda()
    trained_model, train_loss_list, test_loss_list = train(lstm,train_loader,test_loader,10)

    print(train_loss_list)
    print(test_loss_list)

    PATH = '/daintlab/data/sr/'
    torch.save(trained_model, PATH + 'LSTMmodel.pt')