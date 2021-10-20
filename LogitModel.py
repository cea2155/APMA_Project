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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
logisticRegr = LogisticRegression()
linreg = LinearRegression()
from sklearn import metrics
import statsmodels.api as sm

df = pd.read_csv('Consolidated.csv')

df_temp = df.dropna()

regressors = ['Double_Binary_T',
       'Double_Binary_B', 'Single_Binary_B', 'Single_Binary_T', 'F_S_Binary_B', 'F_S_Binary_T',
             'Triple_Binary_T', 'Triple_Binary_B', 'T_Mean_BA', 'T_Mean_SLG', 'B_Mean_BA', 
             'B_Mean_SLG', 'T_ERA', 'T_WHIP', 'B_ERA', 'B_WHIP']



# hyperparameters
# found on https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
# example of grid searching key hyperparametres for logistic regression
# define models and parameters
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2', 'l1', 'elasticnet', 'none']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
logisticRegr = LogisticRegression(penalty = 'l2',
                                  solver='newton-cg', C = .01)

logisticRegr.fit(X_train, y_train)

y_pred = logisticRegr.predict(X_test)

metrics.accuracy_score(y_test, y_pred)


log_reg = sm.Logit(y_train, sm.add_constant(X_train)).fit()
print(log_reg.summary())


X_vals = df_temp[df_temp.Date < datetime.strptime('2020-01-01', '%Y-%m-%d')][regressors]

first_list = [1]
second_list = [1]
for i in X_vals:
    if "Double" in i:
        first_list.append(1)
        second_list.append(0)
    else:
        first_list.append(X_vals[i].median())
        second_list.append(X_vals[i].median())
    
first= 1/(1+(np.exp(-(np.dot(first_list, log_reg.params)))))
second = 1/(1+(np.exp(-(np.dot(second_list, log_reg.params)))))
print((first-second)/(second))