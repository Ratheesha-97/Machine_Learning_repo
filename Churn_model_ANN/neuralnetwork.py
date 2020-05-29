# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:58:41 2020

@author: hp
"""

#artificial neural network using theano ,tensor flow and keras
#theano --- numerical computation forward and backward propogation---parellel--gpu
#tensor flow --- google brain team ---- gpu ---parellel
#keras = theano + tensor flow ---few lines of code ----google ml scientist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("Churn_Modelling.csv")
#data analysis and visualisation

x1 = data.drop(['RowNumber', 'CustomerId', 'Surname','Exited'],axis = 1)
y1 = data['Exited']
x = data.iloc[:,3:13].values
y = data.iloc[:,13].values


print(data.columns)
sample = data.head(5)
print(sample)
corr_matrix = data.corr()
print(corr_matrix)
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix,annot = True)

#encoding the categorical variable
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#there are two categorical variables namely geography and gender
#so we need to create 2 objects of label encoder
#labelencoder_gender = LabelEncoder()
#labelencoder_geography = LabelEncoder()

#x[:,1] = labelencoder_geography.fit_transform(x[:,1])
#x[:,2] = labelencoder_gender.fit_transform(x[:,2])

#since only two categories are there for gender no need of Onehotencoding
#onehotencoder = OneHotEncoder(categorical_features = [1])
#x = onehotencoder.fit_transform(x).toarray()
#x = x[:, 1:]
gender = pd.get_dummies(data['Gender'])
geography = pd.get_dummies(data['Geography'])
new_x = pd.concat([x1,gender,geography],axis = 1)
new_x.drop(['Geography','Gender','Male','Spain'],axis = 1,inplace = True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(new_x,y1,test_size=0.2,random_state = 42) 


#scaling is done to x_train and x_test even for neural network
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#building ANN :) importing the keras library and two modules namely sequential and dense
#sequential ---to initialise the network
#dense ---to build the layers in the network
#models and layers is sublibrary
import keras
from keras.models import Sequential
from keras.layers import Dense 


#initialsing the deep learning model ---ANN
#CREATE AN OBJECT OF SEQUENTIAL CLASS
#our network is a classifier and hence we call our model as classifier

classifier  = Sequential() #defining as sequence of layers



# adding the input layer and the first hidden layer together-----object.method(parameter)

classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=11, units=6))
#Adds a layer instance on top of the layer stack.
#dense--- output = activation(dot(input, kernel) + bias)

#second hidden layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu",units = 6))

#output layer with 1 node and sigmoid function
classifier.add(Dense(kernel_initializer="uniform", activation="sigmoid",units = 1))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#optimizer is the algorithm we use to train the ANN--- adam is a type of stochastic gradient descent
#loss is the optimization criterion like in linear regression ----MSE ---for two class problem it is
#binary_crossentropy and for multi class it is categorical_crossentropy
#metric is the criterion in which every epoch will be improved it is actually a list of criterions

#fiting the ANN model to our data set
classifier.fit(x_train,y_train,batch_size = 10,epochs = 100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
#y_pred is obtained as probabilty of exiting a bank or not

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) #both y_test and y_Pred must of same type i.e 0 or 1




