import pandas as pd
import numpy as np
import torch
from datetime import timedelta, datetime


def loadData(data): # !! it makes first col to row name !!
  df = pd.read_csv(data)
  df = df.set_index('0')

  return df



def getTimeStep(df): # columns to list
  inputList = list(map(int, df.columns.tolist()))
  return inputList



def oneDayIs48(inputList): # detect days which is not 48 timeStep 
                           # and return {day:timeStep} dictionary
  oneDay = 24 * 2 # 24h x 30min
  day = 195
  dayList = []
  tempDic = {}
  for i in range(len(inputList)):
    dayList.append(inputList[i]//100)
  while True:
    if dayList.count(day) != oneDay:
      tempDic[day] = dayList.count(day)
    day += 1
    if day == 731:
      break

  return tempDic


def countOmit(df): # sum(count NAN)
  return sum(df.isnull().sum())



def observe(df,customer,day): # return a day of customer
  dayList = []
  usage = []
  inputList = getTimeStep(df)
  TorF = False
  for i in range(len(inputList)):
    temp = inputList[i]//100
    if temp==day:
      TorF = True
      dayList.append(inputList[i])
      usage.append(df.loc[customer][i])
    elif TorF == True: break

  df = pd.DataFrame(np.array(usage)).transpose()
  df.columns = dayList
  df.index = [customer]
  return df


def findNAN(df):
  inputList = getTimeStep(df)
  NANcolumn = []
    
  for i in range(len(inputList)):
    if df[str(inputList[i])].isnull().values.any()==True:
      NANcolumn.append(inputList[i])

  return NANcolumn



def renameCol(df): # rename columns from Day195 to Day750
  colList = getTimeStep(df)
  colList.remove(73048)
  colList.remove(73047)
  day = 19500
  count = 0
  for i in range(536):
    time = 1
    for j in range(48):
      colList[count] = day + i*100 + time
      time += 1
      count += 1
  
  df = df.drop(columns=['73047','73048'])
  df.columns = colList
  return df


def renameRow(df): # rename index from 1 to 929
  id = [i for i in range(1,929+1)]
  df.index = id
  return df



def getIDtensor(id): # customer id to 31by 2 tensor
  IDtensor = torch.zeros([2,31])
  left = id // 31
  right = id % 31
  IDtensor[0,left] = 1
  IDtensor[1,right] = 1
  return IDtensor



def getMWDtensor(timestep): # returns Month Week Dat facor tensor
  day_0 = datetime(2009,7,1) # day 0 = 1st, July, 2009
  day_target = (timestep//100) - 195
  day = day_0 + timedelta(days=day_target)
  M = day.month 
  W = day.isoweekday() # Monday:Sunday == 1:7
  D = day.day
  MWDtensor = torch.zeros([1,63])
  MWDtensor[0,M-1] = 1
  MWDtensor[0,12+W-1] = 1
  MWDtensor[0,19+D-1] = 1
  return MWDtensor


def getFactorTensor(id,timestep):
  ID = getIDtensor(id).view([1,62])
  MWD = getMWDtensor(timestep)
  factor = torch.cat([ID,MWD],dim=1)
  factor = factor.view([1,5,25])

  return factor



def testFunc(number): # toyCode for testing
  if number<0:
    raise ValueError
  return ("even" if number%2==0 else "odd")



if __name__=="__main__":
  """Test whatever you want"""
  print(getFactorTensor(123,330).shape)
  print(getFactorTensor(123,330))
  pass