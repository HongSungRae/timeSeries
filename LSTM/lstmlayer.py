import torch
import torch.nn as nn
from torchsummary import summary
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import dataProcess


class LSTMLayer(nn.Module):
    def __init__(self,hidden_size=int(7*48*1.5),num_layers=2,**kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size=7*48,hidden_size=hidden_size,num_layers=num_layers,dropout=.5,**kwargs)
        self.fc = nn.Linear(in_features=int(7*48*1.5+1*5*25),out_features=48)

    def forward(self,x,h0_and_c0,factor):
        x, (hn,cn) = self.lstm(x,h0_and_c0)
        print(x.shape)
        factor = factor.reshape(factor.shape[0],-1)
        x = torch.cat([x,factor],dim=1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    lstm = LSTMLayer() # 336,504,2

    dummy_input = torch.zeros(48*7,1,48*7)
    dummy_h0 = torch.zeros(2,1,int(7*48*1.5))
    dummy_c0 = torch.zeros(2,1,int(7*48*1.5))
    dummy_factor = torch.zeros(1,5,25)

    out, (hn,cn) = lstm(dummy_input,(dummy_h0,dummy_c0),dummy_factor)
    print(out)
    summary(lstm,input_size=dummy_input.shape)