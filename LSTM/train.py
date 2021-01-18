import torch.nn.functional as F
import torch.optim as optim
import os
import sys
from model import Model
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from loss import RMSE


def trainlstm(x,y):
    lstm = Model(
        input_size=48*7,
        hidden_size=48,
        num_layers=2
        )

    criterion = RMSE()
    optimizer = optim.Adam(lstm.parameters())

    # start training
    for i in range(5000):
        lstm.train()
        outputs = lstm(x)
        loss = criterion(outputs.view(-1, input_size), y_data.view(-1).long())
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%500 == 0:
            result = outputs.data.numpy().argmax(axis=2)
            result_str = ''.join([char_set[c] for c in np.squeeze(result)])
            print(i, "loss: ", loss.item(), "\nprediction: ", result, "\ntrue Y: ", y_data, "\nprediction str: ", result_str,"\n")


if __name__=='__main__':
    