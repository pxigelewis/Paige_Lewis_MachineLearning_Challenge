#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:01:49 2020

@author: paigelewis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#downloading data

dataframe_all = pd.read_csv('Downloads/1410001701_databaseLoadingData.csv')
num_row = dataframe_all.shape[0]
dataframe_all.head()

#cleaning data
#counting the number of missing elements in each column

counter_nan = dataframe_all.isnull().sum()
counter_without_nan = counter_nan[counter_nan==0]

#removing columns with missing elements

dataframe_all = dataframe_all[counter_without_nan.keys()]


#might have to remove columns, come back to this

x = dataframe_all[:,:-1].values
standard_scaler = StandardScaler()
x_std = standard_scaler.fit_transform(x)

#t distributed stochastic neighbour embeddding

y = dataframe_all.ix[:,-1].values
# encode the class label
class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# step 5: split the data into training set and test set
test_percentage = 0.1
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = test_percentage, random_state = 0)


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state = 0)
x_test_2d = tsne.fit_transform(x_std)


#scatter plot the sample points

markers=('s', 'd', 'o', '^', 'v')
color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan'}
plt.figure()
for idx, cl in enumerate(np.unique(y_test)):
    plt.scatter(x=x_test_2d[y_test==cl,0], y=x_test_2d[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()





