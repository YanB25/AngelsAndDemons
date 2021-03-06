import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
# import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

import trainingdata

def sklearn_clissifiers(trainX, trainY, testX, sss):
    classifier = [
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        LogisticRegression(),
        # SVC(probability=True, gamma=0.001),
    ]

    acc_dict = {}
    times = 0



    for train_index, test_index in sss.split(trainX, trainY):
        times += 1
        print('---- begin round {} ----'.format(times))
        sub_X_train, sub_X_test = trainX[train_index], trainX[test_index]
        sub_Y_train, sub_Y_test = trainY[train_index], trainY[test_index]

        for clf in classifier:
            name = clf.__class__.__name__
            print('training {} ...'.format(name))
            clf.fit(sub_X_train, sub_Y_train)
            train_predictions = clf.predict(sub_X_test)
            auc = roc_auc_score(sub_Y_test, train_predictions)
            if name in acc_dict:
                acc_dict[name] += auc
            else:
                acc_dict[name] = auc

    for clf in classifier:
        name = clf.__class__.__name__
        acc_dict[name] = acc_dict[name] / 10.0

    print(acc_dict)

def lightgbm_clissify(trainX, trainY, testX, sss):
    times = 0
    auc = 0
    for train_index, test_index in sss.split(trainX, trainY):
        sub_X_train, sub_X_test = trainX[train_index], trainX[test_index]
        sub_Y_train, sub_Y_test = trainY[train_index], trainY[test_index]

        lgb_train = lgb.Dataset(sub_X_train, sub_Y_train)
        lgb_eval = lgb.Dataset(sub_X_test, sub_Y_test, reference=lgb_train)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'auc'},
            'num_leaves': 160,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
        }

        gbm = lgb.train(
            params, 
            lgb_train, 
            # learning_rates=lambda iter: 0.1 * (0.995 ** (iter / 10)),
            learning_rates=lambda iter: 0.05,
            num_boost_round=3000, 
            valid_sets=lgb_eval, 
            early_stopping_rounds=100,
            nfold=10,
            verbose_eval=-1)

        print('saving models ...')
        gbm.save_model('model/{}-model.gbm.txt'.format(times))

        y_pred = gbm.predict(sub_X_test, num_iteration=gbm.best_iteration)
        # y_pred_binary = np.where(y_pred >= 0.5, 1, 0)
        auc += roc_auc_score(sub_Y_test, y_pred)

        break
    
    print('auc is {}'.format(auc))


def lightgbm_find_param(trainX, trainY, testX, sss):
    lgb_train = lgb.Dataset(trainX, trainY)
    print('numleaves, auc, std')

    for num_leaves in range(100, 200, 10):
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'auc'},
            'num_leaves': num_leaves,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'boosting': 'dart'
        }

        res = lgb.cv(
            params, 
            lgb_train, 
            num_boost_round=1000, 
            # early_stopping_rounds=100,
            nfold=5,
            verbose_eval=-1)

        auc_mean = np.mean(res['auc-mean'])
        auc_std = np.mean(res['auc-stdv'])
        print(num_leaves, auc_mean, auc_std)


if __name__ == '__main__':
    trainX, testX = trainingdata.getPreProcessed()
    trainY = trainingdata.readTarget()

    trainX = trainX.values
    testX = testX.values
    trainY = trainY.values.reshape(-1)

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

    # sklearn_clissifiers(trainX, trainY, testX, sss)

    # lightgbm_clissify(trainX, trainY, testX, sss)
    lightgbm_find_param(trainX, trainY, testX, sss)

