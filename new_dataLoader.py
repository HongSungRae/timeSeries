import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataProcess import *


class customDataLoader(Dataset):
    def __init__(self, df, purpose):
        # 데이터셋의 전처리를 해주는 부분
        self.df = df
        self.purpose = purpose
        self.target = torch.zeros([1,1,48]).cuda()
        self.end = 0
        if self.purpose=="train":
            # train set. 8일부터 506일까지
            self.df = self.df.iloc[0:,(8-1)*48:506*48]
            self.end = len(self.df.columns)
        else:
            # test set. 마지막 30일. 507일부터 536일(마지막날)
            self.df = self.df.iloc[0:,(507-1)*48:]
            self.end = len(self.df.columns)
    
    def __len__(self):
        # 데이터셋의 길이 즉 샘플의 수를 적어주는 부분
        return len(self.df)
    
    def __getitem__(self,idx):
        # 데이터셋에서 특정 1개의 데이터를 가져오는 함수
        while True:
            start = 0
            start = np.random.randint(start,self.end-8*48) # 적어도 뒤에 8일은 있어야한다
            candidate = self.df.iloc[idx,start:start+48*8]
            if candidate.isnull().values.any()==False: #8일동안 결측치 없으면
                date = candidate.columns[48*4]
                x = torch.zeros([1,7,48]).cuda()
                y = torch.zeros([1,1,48]).cuda()
                for i in range(7):
                    # 7일치 x에 넣어주고
                    pass
                # 하루치 y에 넣어준다
                
                ID = getIDtensor(idx)
                MWD = getMWDtensor(date)
                factor = torch.cat([ID,MWD]) # 5x31 이걸 5x25짜리랑 어떻게 concat할까...
                #!!!factor reshape해줘야합니다!!!
                break
            else:
                continue
 
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
    print("======== input ========\n",x)
    print("======== target ========\n",y)
    print("======== factor ========\n",factor)