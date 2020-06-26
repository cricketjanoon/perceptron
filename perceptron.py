# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:46:17 2020

@author: cricketjanoon
"""

import pandas as pd
import numpy as np

df = pd.read_csv('iris.data', header=None)
df = df.loc[:99] # dropping the third class (only consider two classes)

X = df.loc[:, 0:3].to_numpy() # seperating the features
X = np.hstack((np.ones(shape=(100, 1)), X)) # adding column of 1 for bias term    
Y = np.where(df.iloc[:, -1]=='Iris-setosa', -1, 1) # -1 for 'Iris-setosa' and 1 for 'Iris-versicolor'
    
epochs = 10 # although it only takes 3 epochs to lineraly seperate the data
w_vector = np.zeros((1, X.shape[1])) # initializing the weight matrix

# perceptron learning algorithm
i=0
while i<epochs:    
    misclasified_labels = 0
    for x, y in zip(X, Y):
        if np.dot(y, np.dot(w_vector, x)) <= 0:
            w_vector = w_vector + y*x
            misclasified_labels += 1
    if misclasified_labels == 0:
        print("Breaking the loop. Learning completed in {} iterations.".format(i))
        break
    i+=1

# results, all 'Iris-setosa' have negative results, and 'Iris-versicolor' have positive value
results = np.zeros((100, 1))
for index, x in enumerate(X):
    results[index] = np.dot(w_vector, x)
    
print("Weight vector:", w_vector)