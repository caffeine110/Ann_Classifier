# -*- coding: utf-8 -*-
"""
Spyder Editor

aim : TO build Ann which predicts which employee is likely to leave the job
date :18/6/18

"""

#PART -- 1 :

#Phase - 1 : importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Phase - 2: Prepering Datasets
#importing datasets
dataset = pd.read_csv('data_inp.csv')
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values


#Phase - 3:
#encoding the categerical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


#spliting the data into the training and test datasets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25, random_state = 0)


#Phase - 4 : 
#feature_scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#PART  -- 2 :
#:
#importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


#initialisation of ANN model
classifier = Sequential()

#Adding input layer and one Hiden layer 
classifier.add(Dense(output_dim = 6, init = 'uniform', activation= 'relu', input_dim = 11))

#Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu'))

#Adding the output layer
classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#fitting the ANN to the dataset

classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)



# PART --- 3:

#predicting the test set Results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

#making the confusion Metrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
