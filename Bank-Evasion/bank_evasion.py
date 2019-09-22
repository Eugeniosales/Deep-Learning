#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sun Sep 22 00:55:40 2019

@author: eugenio

Kind of problem: Binary Classification

Problem description: A bank is losing his clients and dont know why

Mission: Figure it out why these clients are leaving and predict a bnary outmcome if either a client will stay or leave

Dataset: 10 mil clients in a period of 6 months serving the information if either the client went away or is still a client
'''

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

## Part 1: Data Pre-processing

#Import the dataset and set the dependent and independent variables
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

#Taking care of missing data

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()

#Encoding the countries
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    
#Encoding the gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#One hot encode the labeled countries
onehotencoder_X = OneHotEncoder(categorical_features = [1])
X = onehotencoder_X.fit_transform(X).toarray()
X = X[:, 1:]

'''
from sklearn.compose import ColumnTransformer 
onehotencoder_X = ColumnTransformer([("Country", OneHotEncoder(),[1])], remainder="passthrough")) # The last arg ([0]) is the list of columns you want to transform in this step
onehotencoder_X.fit_transform(X)  
'''

#Split the dataset beteen test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) 

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


## Part 2: Make the ANN

#Importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Acrchicture of our ANN

#Initializing the ANN as sequence of layers
classifier = Sequential()

#Adding the input layer and the first hidden layer
# dim_output = (12 inputs + 1 output)/2 (this is the number of nodes of the this first hidden layer)
# init: Initialize the weights by small numbers closed to 0 with uniform function
# activation: Rectifier function for the hidden layers
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))

#Adding the secondd hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

##Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#Compiling the ANN
#Optimizer: The algorithm to find the ideal weights (Adam is a kinf of stocahastic gradient descent)
#Loss: Logaritimic loss for the optimizer function
#Accuracy: It refers to the accuracy of the weights. As loss decreases, the accuracy increases
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

## Part 3: Making predictions and evaluating the model

#Fitting the classifier to the Training set
classifier.fit(X_train, Y_train, batch_size=10, nb_epoch=100)

#Predicting the Test set results
Y_pred = classifier.predict(X_test)

##Aplying the threadshold to see use the confusion matrix
Y_pred = (Y_pred >0.5)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print('Accuracy: {}'.format((1557+133)/2000))

## Part 4: Visualizing the results in a chart

## Part 5: Summarizng the conclusions of the study
