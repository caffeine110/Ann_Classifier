#!/usr/bin/env pyth on3 Spyder Editor
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:06:22 2018

@author : Gaurav Gahukar

AIM     : TO build Ann which predicts which Customer is most likely to leave the job
        : implimenting ANN in Sklearn

"""


### importing the libraries
import pandas as pd
import numpy as np


### importing datasets
filePath = 'dataset/raw_data.csv'
dataset = pd.read_csv(filePath)

X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values



### Encoding the categerical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]



### Spliting the data into the training and test datasets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25, random_state = 0)



### Feature_scalling
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

