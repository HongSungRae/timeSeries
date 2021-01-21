import torch
import torch.nn as nn

torch.manual_seed(1)

class Model(nn.Module):
    def __init__(self,hidden_size=256,**kwargs):
        super().__init__()
        self.lstm = nn.LSTM(hidden_size=hidden_size,dropout=.5,**kwargs)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(hidden_size+125,256)
        self.linear2 = nn.Linear(256,48)
    
    def forward(self,x,factor):
        x,_ = self.lstm(x)
        x = self.relu(x)
        x = torch.cat([x,factor],dim=2)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


if __name__=='__main__':
    lstm = Model(
        input_size=48*7,
        hidden_size=256,
        num_layers=5
    )
    
    x = torch.zeros(64,1,48*7)
    factor = torch.randn(64,1,125)
    output = lstm(x,factor)
    print(output.shape)