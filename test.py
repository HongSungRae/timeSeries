import numpy as np
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
  print(total_batch) # 165

  return test_loss_list


if __name__ == "__main__":
  df_test = pd.read_csv("/daintlab/data/sr/testdf.csv",index_col=0)

  test_dataset = myDataLoader(df_test)
  test_loader = DataLoader(test_dataset, shuffle=True, batch_size=64, pin_memory=False)

  print(len(test_loader))
  # model = ??
  test_loss_list = test(model,test_loader)

  print(test_loss_list)