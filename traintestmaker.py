from dataProcess import *
import torch
import pandas as pd
import numpy as np



def mkTraindata(df,dir):
    df = df.iloc[0:,0:506*48]
    df_train = pd.DataFrame(index=[(i//499)+1 for i in range(929*499)],columns=[j for j in range(48*8+1)])
    len_idx = len(df.index)
    len_col = len(df.columns)
    for i in range(len_idx):

    df_train.to_csv(dir,header=True,index=)


def mkTestdata(df,dir):
    df = df.iloc[0:,(507-1)*48:]
    df_test = pd.DataFrame(index=[(i//23)+1 for i in range(929*23)],columns=[j for j in range(48*8+1)])
    len_idx = len(df.index)
    len_col = len(df.columns)
    for i in range(len_idx):

    df_test.to_csv(dir,header=True,index=)

    



if __name__ == "__main__":
    df = loadData("/daintlab/data/sr/df_timeSeries.csv")
    df = renameCol(df) # 536일
    df = renameRow(df)


    '''
    print("==mkTraindata==")
    mkTestdata(df,'/daintlab/data/sr/')
    print("==done==")
    print("==mkTestdata==")
    mkTestdata(df,'/daintlab/data/sr/')
    print("==done==")
    '''