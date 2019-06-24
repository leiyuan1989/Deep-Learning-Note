# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:30:05 2019

@author: leiyuan
"""

from sklearn.datasets import load_iris
iris = load_iris()
a = iris.data
b = iris.target

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

