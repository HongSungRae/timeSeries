import torch
import torch.nn as nn

torch.manual_seed(1)

class Model(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.lstm = nn.LSTM(dropout=.5,**kwargs)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(48+125,48)
    
    def forward(self,x,factor):
        x,_ = self.lstm(x)
        x = self.relu(x) #[480,336,48]
        x = torch.cat([x,factor],dim=2)
        x = self.linear(x)
        return x


if __name__=='__main__':
    lstm = Model(
        input_size=48*7,
        hidden_size=48,
        num_layers=10
    )
    
    x = torch.zeros(929,492,48*7)
    factor = torch.randn(929,492,125)
    output = lstm(x,factor)
    print(output.shape)