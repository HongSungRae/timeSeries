import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataProcess import *


class customDataLoader(Dataset):
    def __init__(self, df, purpose):
        # 데이터셋의 전처리를 해주는 부분
        self.df = df
        self.new_df = ??
        self.purpose = purpose
        self.days = 0
        if self.purpose=="train":
            # train set. 8일부터 506일까지
            self.df = self.df.iloc[0:,(8-1)*48:506*48]
        else:
            # test set. 마지막 30일. 507일부터 536일(마지막날)
            self.df = self.df.iloc[0:,(507-1)*48:]
        self.days = len(self.df.columns)/48
        
        for i in range(len(self.df)):
            for j in range(self.days-8): # 뒤에 8일은 남아있어야한다
                candidate = self.df.iloc[i,j*48:j*48+48*8].to_frame().transpose() 
                if candidate.isnull().values.any() == False:
                    pass
                else: pass
    
    def __len__(self):
        # 데이터셋의 길이 즉 샘플의 수를 적어주는 부분
        return len(self.new_df)
    
    def __getitem__(self,idx):
        factor = getFactorTensor(idx+1,timestep)

        return x, y, factor



if __name__ == "__main__":
    df = loadData("/daintlab/data/sr/df_timeSeries.csv")
    df = renameCol(df)
    df = renameRow(df)

    train_dataset = customDataLoader(df,'train')
    test_dataset = customDataLoader(df,'test')

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64, pin_memory=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64, pin_memory=False)
    
    
    x, y, factor = next(iter(train_loader))
    print("======== input ========\n",x,"\n",x.size())
    print("======== target ========\n",y,"\n",y.size())
    print("======== factor ========\n",factor,"\n",factor.size())


    print("input tensor :",x.size())
    print("target tensor :",y.size())
    print("factor tensor :",factor.size())
    '''

    for batch_idx, samples in enumerate(test_loader):
        x, y, factor = samples
        print(batch_idx)
        print(x)
        print(y)
        print(factor)
        break
    '''