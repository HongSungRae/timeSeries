from dataProcess import *
import torch
import pandas as pd
import numpy as np
import time



def mkTraindata(df,dir):
    df1 = df.iloc[0:,(8-1)*48:(8-1+8)*48]
    for i in range(1,492):
        df2 = df.iloc[0:,(8-1+i)*48:(8-1+i+8)*48]
        df1 = pd.DataFrame(np.concatenate([df1.values, df2.values]), columns=df1.columns)
    df1.columns = [i for i in range(0,384)]
    df1.index = [(j%929)+1 for j in range(0,457068)]
    #col 하나 추가해서 time넣어줘야한다
    df1.to_csv(dir+"/traindf.csv",header=True,index=True)


def mkTestdata(df,dir):
    df1 = df.iloc[0:,(507-1)*48:(507-1+8)*48]
    for i in range(1,23):
        df2 = df.iloc[0:,(507-1+i)*48:(507-1+i+8)*48]
        df1 = pd.DataFrame(np.concatenate([df1.values, df2.values]), columns=df1.columns)
    df1.columns = [i for i in range(0,384)]
    df1.index = [(j%929)+1 for j in range(0,21367)]
    #col 하나 추가해서 time넣어줘야한다
    df1.to_csv(dir+"/testdf.csv",header=True,index=True)


    



if __name__ == "__main__":
    start = time.time()

    df = loadData("/daintlab/data/sr/df_timeSeries.csv")
    df = renameCol(df) # 536일
    df = renameRow(df)
    print('\n\n== df loaded ==\n\n')

    print("==mkTraindata==")
    mkTraindata(df,"/daintlab/data/sr")
    print("==done==")
    print(time.time()-start)

    start = time.time()
    print("==mkTestdata==")
    mkTestdata(df,"/daintlab/data/sr")
    print("==done==")
    print(time.time()-start)

    # df_test = pd.read_csv("/daintlab/data/sr/testdf.csv",index_col=0) 불러올때는 이렇게