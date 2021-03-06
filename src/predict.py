import lightgbm as lgb
import numpy as np
import pandas as pd
from trainingdata import getPreProcessed
from trainingdata import readSubmitTemplate

if __name__ == '__main__':
    bst = lgb.Booster(model_file='model/0-model.gbm.txt')
    _, testX = getPreProcessed()

    y_pred = bst.predict(testX)
    # y_pred_binary = np.where(y_pred >= 0.5, 1, 0)

    submitTemplate = readSubmitTemplate()
    # submitTemplate['label'] = pd.Series(y_pred_binary)
    submitTemplate['label'] = pd.Series(y_pred)

    name = '../data/better-150leveas.lgb.csv'
    submitTemplate.to_csv(name, index=False)
    print('successfully write to {}'.format(name))
