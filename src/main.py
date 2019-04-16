import matplotlib.pyplot as plt
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

classifier = [
    KNeighborsClassifier(),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()
]

acc_dict = {}

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

trainX, testX = trainingdata.getPreProcessed()
trainY = trainingdata.readTarget()

trainX = trainX.values
testX = testX.values
trainY = trainY.values.reshape(-1)

print(trainY.sum())

for train_index, test_index in sss.split(trainX, trainY):
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

for clf in acc_dict:
    name = clf.__class__.__name__
    acc_dict[name] = acc_dict[name] / 10.0

print(acc_dict)