import numpy as np
import pandas as pd
import re as re

# from util import strColumns
from processor import fillEmpty
from processor import string2int
from processor import filterAllSameCols
from processor import split209Datatime
from processor import split247Dash
from processor import dropId
from processor import scaleToStandard

def before_hook(df):
    for col in df.columns:
        tmp = df[col].astype(str)
        t = tmp.str.find('-')
        if np.any(t != -1):
            row = (t != -1).tolist().index(True)
            print(col, row, df.iloc[row, col])
        continue

def readData():
    train = pd.read_csv('../data/train.csv', header=None)
    test = pd.read_csv('../data/test.csv', header=None)
    return train, test

def readTarget():
    trainY = pd.read_csv('../data/label.csv')
    trainY.drop(['Id'], 'columns', inplace=True)
    return trainY

def readTest():
    test = pd.read_csv('../data/test.csv', header=None)
    return test

def readSubmitTemplate():
    return pd.read_csv('../data/submissionExample.csv')


def getPreProcessed():
    try:
        print('load from file')
        pre_processed_train = pd.read_csv('train.preprocess.csv')
        pre_processed_test = pd.read_csv('test.preprocess.csv')
        print(len(pre_processed_test.columns))
        print(len(pre_processed_train.columns))
        return pre_processed_train, pre_processed_test
    except:
        print('calculating.')
        train, test = readData()

        # before_hook(test)

        dataset = [train, test]

        fillEmpty(dataset)
        
        split209Datatime(dataset)
        split247Dash(dataset)
        
        string2int(dataset)
        filterAllSameCols(dataset)

        dropId(dataset)

        scaleToStandard(dataset)

        dataset[0].to_csv('train.preprocess.csv', index=False)
        dataset[1].to_csv('test.preprocess.csv', index=False)

        return tuple(dataset)


if __name__ == '__main__':
    train, test = getPreProcessed()
    print(train)
    print(train.describe())