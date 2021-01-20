import torch
import torch.nn as nn

class RMSEforLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        self.mse = nn.MSELoss()

    def forward(self,y,y_hat):
        return torch.sqrt(self.mse(y,y_hat)+self.eps)


if __name__ == '__main__':
    torch.manual_seed(1)
    y = torch.randn(64,1,48)
    y_hat = torch.randn(64,1,48)

    criterion = RMSEforLSTM()
    loss = criterion(y,y_hat)
    print(loss)