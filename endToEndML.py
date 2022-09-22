#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:21:59 2022

@author: Pranab Mishra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# lets load the boston dataset====================================================
from sklearn.datasets import load_boston

boston = load_boston()

type(boston)
#Out[97]: sklearn.utils.Bunch

boston.keys()
#Out[98]: dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])

# lets check the description
print(boston.DESCR)
print(boston.data)
print(boston.target)
print(boston.feature_names)
print(boston.filename)

# lPrepare the boston dataset======================================================

dataset = pd.DataFrame(boston.data,columns = boston.feature_names)
dataset.head()
dataset['PRICE'] = boston.target

## to set no of rows and columns for display
pd.get_option('display.max_rows')
pd.get_option('display.max_columns')
pd.set_option('display.max_columns' ,None)
pd.set_option('display.max_rows' ,None)

dataset.shape

dataset.info()

#summarizing the stats of data
dataset.describe()

#missing value check
dataset.isnull().sum()

#EDA
##corelation > super important for any regression problem
## perason corealtion checks the relation between 2 variable between -1 to 1
dataset.corr()

sns.heatmap(dataset,vmin = 0,vmax = 1,center =0)
sns.heatmap(dataset,center =0,annot=True)

ax = sns.heatmap(dataset, cmap="YlGnBu",annot=True)

sns.pairplot(dataset)


plt.scatter(dataset['CRIM'],dataset['PRICE'])
plt.xlabel('Crime Rate')
plt.ylabel('Price')

plt.scatter(dataset['RM'],dataset['PRICE'])
plt.xlabel('RM')
plt.ylabel('Price')

sns.regplot(x = 'RM', y = 'PRICE', data = dataset)
sns.regplot(x = 'CRIM', y = 'PRICE', data = dataset)
sns.regplot(x = 'LSTAT', y = 'PRICE', data = dataset)
sns.regplot(x = 'CHAS', y = 'PRICE', data = dataset)
sns.regplot(x = 'PTRATIO', y = 'PRICE', data = dataset)

#create model
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
X.head()
y.head()

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size =0.3,random_state = 42)

X_train.shape
X_train.head()
y_train.shape
y_train.head()

X_test.shape
X_test.head()
y_test.shape
y_test.head()

#scaling dataset
#we have to normalize or standardize to reach uniform scaling
#internally we use Gradient Descent and our main aim is to reach the lobal minima
# so all our independent variables shoul be of same scale - the convergence happens quickly

from sklearn.preprocessing  import StandardScaler
scale = StandardScaler()

X_train = scale.fit_transform(X_train)

X_test= scale.transform(X_test)

#Train our model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train,y_train)

# print the coefficient of intercept
print(lr.coef_)

print(lr.intercept_)

#On which parameters the model has been trained
lr.get_params()

# Prediction with test data
reg_pred = lr.predict(X_test)

##plot for predcition

plt.scatter(y_test,reg_pred)

## prediction with residuals

#claculate residual

residuals = y_test-reg_pred

sns.displot(residuals,kind = 'kde')
# residuals are normally distributed ######################################

## scatter plot wrt prediction and residuals
plt.scatter(reg_pred,residuals)

## unifor distribution between prediction and residuals ####################

from sklearn.metrics  import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_squared_error(y_test,reg_pred))
print(mean_absolute_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))

## R square and adjusted r square
from sklearn.metrics import r2_score
score = r2_score(y_test,reg_pred)

#0.7112260057484932#

##adjusteed Rsquare
X_test.shape
1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

#0.6840226584639308#


# new data prediction

boston.data[0].reshape(1,-1)

## never forget standardization====================================================


scale.transform(boston.data[0].reshape(1,-1))
lr.predict(scale.transform(boston.data[0].reshape(1,-1)))

## picking the model for deployment
import pickle

pickle.dump(lr,open('regmodel.pkl','wb'))

pickled_model = pickle.load(open('regmodel.pkl','rb'))

pickled_model.predict(scale.transform(boston.data[0].reshape(1,-1)))










































