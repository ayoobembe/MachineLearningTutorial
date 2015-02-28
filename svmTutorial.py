# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 14:17:44 2015

@author: danielayomikun
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.00001, C=100)

x,y = digits.data[:-1], digits.target[:-1]
clf.fit(x,y)

print('Prediction:', clf.predict(digits.data[-2]))

plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()


