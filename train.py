import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import loadLayer



def train(??):

    net = LoadCNN().cuda()
    optimizer = optim.SGD(net.parameters(),lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    epochs = 25
    total_batch = len(train_dataloader)

    train_loss_list = []
    val_loss_list = []