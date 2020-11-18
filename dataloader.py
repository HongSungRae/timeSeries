import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataProcess import *


class myDataLoader(Dataset):
    def __init__(self,df):
        self.df = df
        self.len_idx = 0
        self.id_list = []
        self.timestep_list = []

        #결측치 있는 행 모두 제거
        self.df = self.df.dropna(axis=0)

        self.id_list = self.df.index
        self.timestep_list = list(self.df.iloc[0:,-1].values)

    def __len__(self):
        return(len(self.df.index))

    def __getitem__(self,index):
        x = torch.tensor(self.df.iloc[index,0*48:7*48]).view(1,7,48)
        y = torch.tensor(self.df.iloc[index,7*48:8*48]).view(1,1,48)
        factor = getFactorTensor(self.id_list[index],self.timestep_list[index])

        return x, y, factor



if __name__ == "__main__":
    df_test = df_test = pd.read_csv("/daintlab/data/sr/testdf.csv",index_col=0)
    
    test_dataset = myDataLoader(df_test)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64, pin_memory=False)

    x, y, factor = next(iter(test_loader))

    print("x :",x.shape)
    print("y :",y.shape)
    print("factor :",factor.shape)