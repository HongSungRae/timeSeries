#########################################
### This is a dummy code for practice ###
#########################################

import pandas as pd
import numpy as np
import torch

from dataProcess import *


df = loadData("/daintlab/data/sr/df_timeSeries.csv")


df = renameCol(df)
df = renameRow(df)
#Hi

print(df.head())
print(df.iloc[0:,534*48:])

print(df.iloc[0:,534*48:].columns[1])

'''
for i in df.index:
    if df.loc[i,'24601':'24648'].isnull().values.any()==True:
        print(i)
        print(df.loc[i,'24601':'24648'])
        print('=================')
    else:
        pass
'''
#print(df.loc[:,'24601':'24648'].isnull().values.any())


#print(getIDtensor(1))
#print(8064/64)

'''
a = torch.zeros([2,64,5,25])
b = torch.zeros([2,2,5,25])
print(torch.cat([a,b],dim=1).size())

def summ(a,b=10):
    return a+b

print('One',summ(2,4))
print('Two',summ(a=3))
'''

