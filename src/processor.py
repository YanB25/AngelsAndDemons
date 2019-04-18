import numpy as np
import pandas as pd
from sklearn import preprocessing

from util import strColumns
from util import allSameCol

def fillEmpty(dataset):
    print("fill NaN to their mean ...")
    for data in dataset:
        for col in data.columns:
            try:
                mean = data[col].fillna(0).mean()
                print('column {} fill to mean {}'.format(col, mean))
                data[col].fillna(mean, inplace=True)
            except:
                mean = data[col].value_counts().index[0]
                print('column {} not able to fill to mean. fall back to most occurance{}'.format(col, mean))
                data[col].fillna(mean, inplace=True)

        # data.fillna(0, inplace=True)

def split209Datatime(dataset):
    print('(col 209) spliting data time yyyy-mm-dd-hh.mm.ss.000000 ...')
    for df in dataset:
        dt_col = df[209] # the 209th column is to be split

        def getYear(s):
            return int(s.split('-')[0])
        def getMonth(s):
            # print('s is {}, split is {}'.format(s, s.split('-')[1]))
            return int(s.split('-')[1])
        def getDay(s):
            return int(s.split('-')[2])
        def getHour(s):
            return int(s.split('-')[3].split('.')[0])
        def getMin(s):
            return int(s.split('-')[3].split('.')[1])
        def getSec(s):
            return int(s.split('-')[3].split('.')[2])
        
        df['year'] = dt_col.apply(getYear)
        df['month'] = dt_col.apply(getMonth)
        df['day'] = dt_col.apply(getDay)
        df['hour'] = dt_col.apply(getHour)
        df['minute'] = dt_col.apply(getMin)
        df['second'] = dt_col.apply(getSec)

        df.drop([209], 'columns', inplace=True)

def split247Dash(dataset):
    '''
    split 247th column into partA, partB and partC
    247th column is like `%-%-%`
    some colunms are `UNKNOWN`, some are `NaN` and has been converted to 0 in previous steps.
    so `UNKNOWN` and `0` are the same.
    '''
    print('(col 247) spliting aa-bb-cc into part A, B and C ...')
    for df in dataset:
        target_col = df[247]

        def partA(s):
            try:
                if s == 'UNKNOWN':
                    return '?'
                if s == 0:
                    return '?'
                return s.split('-')[0]
            except:
                print('{} can not handle'.format(s))
                return ''

        def partB(s):
            try:
                if s == 'UNKNOWN':
                    return '?'
                if s == 0:
                    return '?'
                return s.split('-')[1]
            except:
                print('{} can not be handled'.format(s))
                return ''
        def partC(s):
            try:
                if s == 'UNKNOWN':
                    return '?'
                if s == 0:
                    return '?'
                return s.split('-')[2]
            except:
                print('{} can not be handled'.format(s))
                return ''
        
        df['PartA'] = target_col.apply(partA)
        df['PartB'] = target_col.apply(partB)
        df['PartC'] = target_col.apply(partC)

        df.drop([247], 'columns', inplace=True)

def string2int(dataset):
    '''
    string2int convert string fields in df to int.
    '''
    print('convert all string fields to int ...')
    s = strColumns(dataset[0])
    for colIdx in s:
        names = {}
        last = 0
        partial = dataset[0][colIdx]
        for name in partial:
            if name not in names:
                names[name] = last
                last += 1
    
        def convertor(name):
            try:
                return int(names[name])
            except:
                print('[WARNING] {} not in converting dict'.format(name))
                return -1
        for df in dataset:
            df[colIdx] = df[colIdx].apply(convertor)
    # print(df)

def dropId(dataset):
    print('(col 0) dropping first colunm, id ...')
    for df in dataset:
        df.drop([0], 'columns', inplace=True)
        

def filterAllSameCols(dataset):
    print('filter out all-same columns ...')
    azc = allSameCol(dataset[0])
    for data in dataset:
        azc_cur = allSameCol(data)
        azc = np.logical_and(azc, azc_cur)
    azci = azc.index[azc] # a list of index that azc is true
    for data in dataset:
        data.drop(azci, 'columns', inplace=True)

def scaleToStandard(dataset):
    print('stardardize to 0-mean and 1-var ...')
    ret = []
    for idx, df in enumerate(dataset):
        val = df.values
        # scale to [0, 1]
        min_max_scaler = preprocessing.MinMaxScaler()
        val_range = min_max_scaler.fit_transform(val)

        # scale to 0-means and 1-std
        val_scaled = preprocessing.scale(val_range)
        dataset[idx] = pd.DataFrame(val_scaled, columns=df.columns)




# 209列是日期
# 247列是 a-b-c型数据