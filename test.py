### 모르겠써요 train하면서 test set한번씩 넣으면 되는데 이걸 왜 만든거지 ###


mport numpy as np
import pandas as pd
import torch
from dataloader import myDataLoader
from torch.utils.data import Dataset, DataLoader
from loss import *


def test(model,test_loader):
  is_cuda = torch.cuda.is_available()
  device = torch.device('cuda' if is_cuda else 'cpu')

  model = model
  test_loss_list = []

  total_batch = len(test_loader)
  print(total_batch) # 329

  for eph in range(epochs):
    loss_learning = 0.0
    for i,data in enumerate(test_loader):
        x, target, factor = data
        if is_cuda:
          x = x.float().cuda()
          target = target.float().cuda()
          factor = factor.float().cuda()

  return test_loss_list


if __name__ == "__main__":
  df_test = pd.read_csv("/daintlab/data/sr/testdf.csv",index_col=0)

  test_dataset = myDataLoader(df_test)
  test_loader = DataLoader(test_dataset, shuffle=True, batch_size=64, pin_memory=False)

  # print(len(test_loader))
  # model = ??
  test_loss_list = test(model,test_loader)

  print(test_loss_list)