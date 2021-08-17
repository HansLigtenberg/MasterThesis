"""
ReadData.py

Purpose: Reads, transforms and saves data

Version:
    1   Read data and select the correct value premium
    2   Collect all data in one data frame
Date: 20210318

Author: Hans Ligtenberg
"""

### Imports
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import scipy as sc

#Functions
def stepSum(array, step):
    return array[::step].sum()

def DEF(df):
    """Dataframe with default premium from Goyal's data"""
    return df['BAA']-df['AAA']

def TERM(df):
    """Dataframe with term premium from Goyal's data"""
    return df['tbl']-df['lty']

def RREL(df):
    """Dataframe with stochastically detrended Rf from Goyal's data"""
    return df['Rfree']-df['Rfree'].rolling(12).mean().shift()

def DP(df):
    """Dataframe with not logged dividend price-ratio from Goyal's data"""
    return df['D12']/df['Index']

def PE(df):
    """Dataframe with price-earnings from Goyal's data"""
    return df['Index']/df['E12'].rolling(120).apply(stepSum, args=tuple([12]))

def VOL(df):
    """Dataframe with realised from Goyal's data"""
    return df['svar']

def readFF(fileName='data/FF.csv'):
    """
    Purpose: Read the Fama French file
    
    Inputs: fileName of csv file
    
    Return value: dataframe with market excess return and Rf indexed by date
    """
    df=pd.read_csv(fileName)
    df=df.rename(columns={'Unnamed: 0':'date',
                          'Mkt-RF':'M',})
    df['date']=pd.to_datetime(df['date'],format='%Y%m')
    df=df.set_index('date')

    return df[['M','RF','HML']]

def readBM(fileName='data/BM.csv'):
    """
    Purpose: Read the book-to-market file
    
    Inputs: fileName of csv file
            sort=decile, quintile, triple
    
    Return value: dataframe with value premia indexed by date
    """
    df=pd.read_csv(fileName)
    df=df.rename(columns={'Unnamed: 0':'date'})
    df['date']=pd.to_datetime(df['date'],format='%Y%m')
    df=df.set_index('date')
    df['VP 10']=df['Hi 10']-df['Lo 10']
    df['VP 20']=df['Hi 20']-df['Lo 20']
    df['VP 30']=df['Hi 30']-df['Lo 30']
    return df[['VP 10', 'VP 20', 'VP 30']]

def readGoyal(fileName='data/Goyal.xlsx', variables=[DEF, TERM, RREL, DP, PE, VOL]):
    """
    Purpose: Read the Goyal file and transforms
    
    Inputs: fileName of excel file
            list of functions for variables to keep 
    
    Return value: dataframe with variables indexed by date
    """
    df=pd.read_excel(fileName)
    df=df.rename(columns={'yyyymm':'date'})
    df['date']=pd.to_datetime(df['date'],format='%Y%m')
    df=df.set_index('date')

    for func in variables:
        df[func.__name__]=func(df)

    return df[[f.__name__ for f in variables]]

def readINDPRO(fileName='data/INDPRO.csv'):
    """
    Purpose: Read the INDPRO file and transform to year on year log change
    
    Inputs: fileName of excel file
                
    Return value: dataframe with variables indexed by date
    """
    df=pd.read_csv(fileName)
    df=df.rename(columns={'DATE':'date', 'INDPRO':'IP'})
    df['date']=pd.to_datetime(df['date'], format='%Y-%m-%d')
    df=df.set_index('date')
    df['IP']=np.log(df['IP'].shift(12)-np.log(df['IP']))
    return df

def readUNRATE(fileName='data/UNRATE.csv'):
    """
    Purpose: Read the UNRATE file and transform to year on year log change
    
    Inputs: fileName of excel file
                
    Return value: dataframe with variables indexed by date
    """
    df=pd.read_csv(fileName)
    df=df.rename(columns={'DATE':'date', 'UNRATE':'UE'})
    df['date']=pd.to_datetime(df['date'], format='%Y-%m-%d')
    df=df.set_index('date')
    df['UE']=np.log(df['UE'].shift(12)-np.log(df['UE']))
    return df

def readCPIAUCSL(fileName='data/CPIAUCSL.csv'):
    """
    Purpose: Read the CPIAUCSL file and transform to year on year log change
    
    Inputs: fileName of excel file
                
    Return value: dataframe with variables indexed by date
    """
    df=pd.read_csv(fileName)
    df=df.rename(columns={'DATE':'date', 'CPIAUCSL':'INF'})
    df['date']=pd.to_datetime(df['date'], format='%Y-%m-%d')
    df=df.set_index('date')
    df['INF']=np.log(df['INF'].shift(12)-np.log(df['INF']))
    return df

def mergeDF(dfList):
    """"Combine the given data frames"""
    df=pd.DataFrame(dfList[0])
    for dfa in dfList[1:]:
        df=pd.merge(df,dfa, how='outer',left_index=True, right_index=True)
        
    return df    

def data(variables=['VP 10', 'VP 20','VP 30','HML', 'M', 'RF', 'DEF', 'TERM', 'RREL', 'DP', 'PE', 'VOL', 'INF', 'UE', 'IP'],
         fileNameBM='data/BM.csv',
         fileNameGoyal='data/Goyal.xlsx',
         fileNameCPIAUCSL='data/CPIAUCSL.csv',
         fileNameUNRATE='data/UNRATE.csv',
         fileNameINDPRO='data/INDPRO.csv',
         fileNameFF='data/FF.csv'):
    """
    Purpose: Gives data frame with in addition to the value premium the specified variables
    
    Inputs: variables=list of strings of variables to include
            locations of files        
        
    Return value: dataframe with variables indexed by date
    """

    dfList=[readBM(fileNameBM),readFF(fileNameFF),readGoyal(fileNameGoyal),
            readCPIAUCSL(fileNameCPIAUCSL),readUNRATE(fileNameUNRATE),
            readINDPRO(fileNameINDPRO)]
    dfList[0]=dfList[0].rename(columns={'HML':'VP'})
    df=mergeDF(dfList)
    return df[variables]
    
def saveData(location='data/df',
         variables=['VP 10', 'VP 20','VP 30','HML', 'M', 'RF', 'DEF', 'TERM', 'RREL', 'DP', 'PE', 'VOL', 'INF', 'UE', 'IP'],
         fileNameBM='data/BM.csv',
         fileNameGoyal='data/Goyal.xlsx',
         fileNameCPIAUCSL='data/CPIAUCSL.csv',
         fileNameUNRATE='data/UNRATE.csv',
         fileNameINDPRO='data/INDPRO.csv',
         fileNameFF='data/FF.csv'):
    """Reads in data and saves as pickle at given location."""
    df=data(variables,fileNameBM,fileNameGoyal,fileNameCPIAUCSL,fileNameUNRATE,fileNameINDPRO,fileNameFF)
    df.to_pickle(location)
    return

def loadData(location='data/df'):
    """Loads the data in location as data frame."""
    return pd.read_pickle(location)
    