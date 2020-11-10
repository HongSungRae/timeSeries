import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataProcess import *

class Drawer():
    def __init__(self,df,batch_size):
        self.df = df
        self.batch_size = batch_size
        self.id = 0
        self.id_list = []
        self.inputTensor = torch.zeros([self.batch_size,7,48])
        self.targetTensor = torch.zeros([self.batch_size*1*48]) # input은 flatten 해서 dense layer 통과후에 loss 구할거니까 미리 flatten해두자
        # self.factor

    def dtloader(self,usage):
        period = [701 if usage=='test' else 202, 730 if usage=='test' else 700]
        while True:
            self.id = np.random.randint(1,929+1) # 고객하나에 대해서
            if self.id in self.id_list:
                continue
            else: self.id_list.append(self.id)
            length = len(self.id_list)

            while True:
                start_day = np.random.randint(period[0],period[1]-6)
                df_8days = df.loc[self.id,((start_day*100)+1):start_day*100+748] # target까지 뽑는건 8일치를 뽑는다
                if df_8days.isnull().values.any() == True: # 시계열에 결측치 있으면 다시 뽑는다
                    continue
                else: # 아니라면 7일치를 tensor에 저장
                    s = 0
                    e = 48
                    for i in range(8):
                        for j in range(s,e):
                            self.inputTensor[length,i,np.asarray(df_8days)[j]] # 외 않돼는지 몰르갯다
                        s += 48
                        e += 48
                    self.targetTensor[(length-1)*48:length*48] = np.asarray(df_8days)[336:384]
                    break

                    

            # timstep으로 월 일 주 factor추출(앞의 7일만). 후에 tensor로 만듦
            # id tensor로 만듦
            # 위 둘을 concat
            
            if len(self.id_list)==self.batch_size:
                break
            else: pass
        # 8일차를 target으로
        # 나머지는 input data로
        # input data, target, factor를 return
        #return data,target,factor
        return self.inputTensor, self.targetTensor #,self.factor



if __name__ == "__main__":
    df = loadData("data/df_timeSeries.csv")
    df = renameRow(renameCol(df))

    loader = Drawer(df,64)
    #train_data, train_target, train_factor = loader.dtloader(train)
    #test_data, test_target, test_factor = loader.dtloader(test)
    #val_data, val_target, val_factor = laoder.dtloader(val)

    inputTensor, targetTensor = loader.dtloader('train')
    print('=======▼input=======',inputTensor,'\n','=======▼target=======')
    print('\n','=======▼customer ID=======',loader.id_list)