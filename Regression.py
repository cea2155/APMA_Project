#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 21:38:29 2021

@author: christineanagnos
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
linreg = LinearRegression()
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pd.read_csv('Consolidated.csv')


df['Date'] = pd.to_datetime(df.Date)
df['Score_Diff'] = [np.abs(int(i.split('-')[0])-int(i.split('-')[1])) for i in df.Score]
init_cond = [
    df.Score_Diff == 0,
    df.Score_Diff != 0
]
init_vals = [0, 1]
df['Initial_Discrepancy'] = np.select(init_cond, init_vals)

df.Discrepancy.value_counts()

df_temp = df.dropna()


regressors = ['Score_Diff',  'Initial_Discrepancy', 'Double_Binary_B', 'Double_Binary_T', 'Single_Binary_B', 'Single_Binary_T',
              'F_T_Binary_B', 'F_T_Binary_T', 
              'Triple_Binary_B', 'Triple_Binary_T', 'S_T_Binary_B', 'S_T_Binary_T',
              'Loaded_Binary_B', 'Loaded_Binary_T',  'T_Mean_BA', 'T_Mean_SLG', 'B_Mean_BA', 
            'B_Mean_SLG', 'T_ERA', 'T_WHIP', 'B_ERA', 'B_WHIP'
    ]


X = df_temp[regressors]
y = df_temp['Discrepancy']

X_train = df_temp[df_temp.Date < datetime.strptime('2020-01-01', '%Y-%m-%d')][regressors]
y_train = df_temp[df_temp.Date < datetime.strptime('2020-01-01', '%Y-%m-%d')].Discrepancy
X_test = df_temp[df_temp.Date > datetime.strptime('2020-01-01', '%Y-%m-%d')][regressors]
y_test = df_temp[df_temp.Date > datetime.strptime('2020-01-01', '%Y-%m-%d')].Discrepancy
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

# hyperparameters
# found on https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
# example of grid searching key hyperparametres for logistic regression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

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

print(metrics.accuracy_score(y_test, y_pred))

print(metrics.recall_score(y_test, y_pred))

import statsmodels.api as sm

reg_sm = sm.Logit(y_train, X_train).fit()

print(reg_sm.summary())


X_vals = df_temp[df_temp.Date < datetime.strptime('2020-01-01', '%Y-%m-%d')][regressors]
param_list = []
new_param_list = []
for a in regressors:
    if 'Double' in a:
        param_list.append(1)
        new_param_list.append(0)
    else:
        param_list.append(X_vals[a].median())
        new_param_list.append(X_vals[a].median())
init_perc = 1/(1+ np.exp(-(np.dot(logisticRegr.coef_, param_list))))[0]
new_perc = 1/(1+ np.exp(-(np.dot(logisticRegr.coef_, new_param_list))))[0]
effect = init_perc-new_perc
print('Effect of Double Binaries: {}'.format(effect))



control_df = df_temp[(df_temp.Date > datetime.strptime(
    '2020-01-01', '%Y-%m-%d')) & (df_temp.Inning > 9)]


control_df['Double_Binary_B'] = 0
control_df['Double_Binary_T'] = 0


X_test_new = control_df[regressors]
X_test_new = scale.fit_transform(X_test_new)
y_pred_new = logisticRegr.predict(X_test_new)
control_df['New_Pred'] = y_pred_new

count_df = pd.DataFrame(control_df.New_Pred.value_counts(), 
                       )
count_df['Discrepancy'] = control_df.Discrepancy.value_counts()


ax = plt.subplot(1,1,1)
width = 0.2
ax.bar(count_df.index, count_df.Discrepancy, 
       width, align = 'center', linewidth = 0.02)
ax.bar(count_df.index+width, count_df.New_Pred, 
       width, align = 'center', linewidth = 0.02)
ax.bar(count_df.index+width, count_df.Predicted, 
       width, align = 'center', linewidth = 0.02)
ax.set_xticks(count_df.index + width / 2)
ax.set_xticklabels((count_df.index))
ax.legend((['Actual Run Discrepancy', 'Simulated Run Discrepancy']), loc = 'lower left')
ax.set_title('Post 2020 Extra Inning Run Discrepancies')





