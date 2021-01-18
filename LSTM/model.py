import torch
import torch.nn as nn

torch.manual_seed(1)

class Model(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.lstm = nn.LSTM(dropout=.5,**kwargs)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(48,48)
    
    def forward(self,x):
        x,_ = self.lstm(x)
        x = self.relu(x) #[929,480,48]
        #x = torch.cat([x,factor])
        x = self.linear(x)
        return x


if __name__=='__main__':
    lstm = Model(
        input_size=48*7,
        hidden_size=48,
        num_layers=2
    )
    
    x = torch.randn([929,480,48*7])
    y = torch.randn([929,480,48])
    factor = torch.randn([1,5,25])
    output = lstm(x)
    print((output-y).shape)