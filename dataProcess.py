import pandas as pd
import numpy as np
import torch


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
  IDtensor = torch.zeros([31,2])
  left = id // 31
  right = id % 31
  IDtensor[left,0] = 1
  IDtensor[right,1] = 1
  return IDtensor




def testFunc(number): # toyCode for testing
  if number<0:
    raise ValueError
  return ("even" if number%2==0 else "odd")