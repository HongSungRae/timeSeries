import pandas as pd
import numpy as np

from dataProcess import *


df = loadData("/daintlab/data/sr/df_timeSeries.csv")

#df = renameCol(df)
#df = renameRow(df)
#Hi

print(df.head())
print(df.loc[0:][24601:24648])


#print(getIDtensor(1))
#print(8064/64)