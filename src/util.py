def strColumns(df):
    '''
    return a list of int indicating indexes, in which columns represent string.
    '''
    return [col for col, dt in df.dtypes.items() if dt == object]

def allSameCol(df):
    return df.describe().loc['min', :] == df.describe().loc['max', :]