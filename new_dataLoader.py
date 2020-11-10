import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataProcess import *


class customDataLoader(Dataset):
    def __init__(self, df, purpose):
        # df는 날짜로 columns 구간을 정해줘서 넣어야한다
        # 그니까 input으로 들어오는 df
        self.candidate = 랜덤하게 8일
        self.y =
        pass
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self):
        return x,factor,y