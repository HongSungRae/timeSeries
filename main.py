import time
import numpy as np
import pandas as pd
import torch
from loadLayer import *
from new_dataLoader import customDataLoader

if __name__ == "__main__":
    df = loadData("/daintlab/data/sr/df_timeSeries.csv")
    df = renameCol(df)
    df = renameRow(df)

    net = LoadCNN().cuda()
    
    trainDataLoader = customDataLoader(df,'train')
    testDataLoader = customDataLoader(df,'test')
