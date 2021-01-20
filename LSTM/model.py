import torch
import torch.nn as nn

torch.manual_seed(1)

class Model(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.lstm = nn.LSTM(dropout=.5,**kwargs)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(256+125,48)
    
    def forward(self,x,factor):
        x,_ = self.lstm(x)
        x = self.relu(x) #[480,336,48]
        x = torch.cat([x,factor],dim=2)
        x = self.linear(x)
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