#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 13:27:26 2021

@author: christineanagnos
"""

import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
logisticRegr = LogisticRegression()
linreg = LinearRegression()
from sklearn import metrics

df = pd.read_csv('Consolidated.csv')

df_temp = df.dropna()

regressors = ['Double_Binary_T',
       'Double_Binary_B', 'Single_Binary_B', 'Single_Binary_T', 'F_S_Binary_B', 'F_S_Binary_T',
             'Triple_Binary_T', 'Triple_Binary_B', 'T_Mean_BA', 'T_Mean_SLG', 'B_Mean_BA', 
             'B_Mean_SLG', 'T_ERA', 'T_WHIP', 'B_ERA', 'B_WHIP']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

logisticRegr.fit(X_train, y_train)

y_pred = logisticRegr.predict(X_test)

metrics.accuracy_score(y_test, y_pred)