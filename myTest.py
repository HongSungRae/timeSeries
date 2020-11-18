#########################################
### This is a dummy code for practice ###
#########################################


import pandas as pd
import numpy as np
import torch

from dataProcess import *

'''
df = loadData("/daintlab/data/sr/df_timeSeries.csv")


df = renameCol(df)
df = renameRow(df)
#Hi

'''

df = pd.DataFrame(
                [ [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]],
                index = [1, 2, 3]
                )
print(df)
print(type(df))


new_df = pd.DataFrame([[]])
print(new_df)
temp_df = pd.DataFrame([[0 for i in range(48*8+2)]])
print(new_df.append(temp_df,ignore_index=True))