# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:11:05 2019

@author: leiyuan
"""

import numpy as np
from sklearn.model_selection import KFold

t = list(range(10))
X = np.array(['x_' + str(t) for t in t])
y = np.array(['y_' + str(t) for t in t])

kf = KFold(n_splits=3)
for train, test in kf.split(X):
    print("%s %s" % (train, test))
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]    
    print( X_train, X_test, y_train, y_test)

from sklearn.model_selection import RepeatedKFold
random_state = 12883823
rkf = RepeatedKFold(n_splits=2, n_repeats=3, random_state=random_state)
for train, test in rkf.split(X):
    print("%s %s" % (train, test))

   
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
for train, test in loo.split(X):
    print("%s %s" % (train, test))

from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=5, test_size=0.2,
    random_state=0)
for train_index, test_index in ss.split(X):
    print("%s %s" % (train_index, test_index))


from sklearn.model_selection import StratifiedKFold

X = np.ones(10)
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    print("%s %s" % (train, test))


from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=4)
print(tscv)  

for train, test in tscv.split(X):
    print("%s %s" % (train, test))



