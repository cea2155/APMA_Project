#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 22:04:11 2021

@author: christineanagnos
"""

import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
import matplotlib as plt
from numpy.linalg import inv
from numpy.linalg import matrix_power
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('CleanedData.csv')


absorbing_states = sorted([i for i in df.State_Out_Combo.unique() if '-3' in i])
transient_states = sorted([i for i in df.State_Out_Combo.unique() if i 
                   not in absorbing_states])

df['Date'] = pd.to_datetime(df.Date)

df_use = df[df.Date < datetime.strptime('2020-01-01', '%Y-%m-%d')]

combo_list = sorted([i for i in df_use.State_Out_Combo.unique().tolist()
                    if '-3' not in i])
combo_list.append('0-3')
combo_list.append('1-3')
combo_list.append('2-3')
combo_list.append('3-3')
transition_list = df_use.Transition.dropna().tolist()
first_state_list = [i[:3] for i in transition_list]

count_df = pd.DataFrame()
start_state = []
count_start_state = []
for i in combo_list:
    start_state.append(i)
    count_start_state.append(first_state_list.count(i))
count_df['Start_State'] = start_state
count_df['Total_Count'] = count_start_state

display(count_df.head())

for i in combo_list:
    count_df['{}'.format(i)] = np.nan

count_df = count_df.set_index('Start_State')

for i in count_df.index:
    total_count_temp = int(count_df.loc[[i]].Total_Count)
    for a in count_df.index:
        if total_count_temp > 0:
            count_i_to_a = transition_list.count('{} to {}'.format(i, a))
            perc = count_i_to_a/total_count_temp
            count_df.at[i,a] = perc
        if total_count_temp == 0:
            count_df.at[i,a] = 0
    if i in [i for i in ['0-3', '1-3', '2-3', '3-3']]:
        count_df.at[i,i] = 1

transition_matrix = count_df.drop(columns = ['Total_Count'])
transition_matrix.to_csv('transition_matrix.csv')

power_example = pd.DataFrame()
for i in range(1,12):
    temp_df = pd.DataFrame(matrix_power(transition_matrix.as_matrix(), i), 
                        columns = transition_matrix.columns
                       ).set_index(transition_matrix.index)
    
    power_example['P^{}'.format(i)] = temp_df.loc['0-1'].tolist()

power_example = power_example.set_index(transition_matrix.columns)
    

display(power_example)



transient = transition_matrix.iloc[:24, :24]

#generate identity matrix:
id_matrix = np.identity(24)

subtracted_matrix = pd.DataFrame(id_matrix, columns = 
        transient.columns).set_index(
    transient.index).sub(transient)

test = subtracted_matrix.as_matrix()

inverse = inv(test)

N_df = pd.DataFrame(
    inverse, columns = transient.columns).set_index(
    transient.index)

display(N_df)

R_df = transition_matrix[absorbing_states].loc[transient_states]

display(R_df)

B_df = pd.DataFrame(np.matmul(N_df.as_matrix(), R_df.as_matrix())).set_index(
    transient.index)

display(B_df)

EL_list = []
ER_list = []
EB_list = []
index = []
for i in B_df.index:
    temp_EL = sum([a*B_df.loc[i][a] for a in range(0, 4)])
    EL_list.append(temp_EL)
    temp_EB = np.sum(N_df.loc[i])
    EB_list.append(temp_EB)
    outs_left = 3 - (int(i.split('-')[1]))
    
    state = int(i.split('-')[0])
    
    if state == 0:
        MoB = 0
    
    if state == 1 or state == 2 or state == 4:
        MoB = 1
    
    if state == 3 or state == 5 or state == 6:
        MoB = 2
    
    if state == 7:
        MoB = 3
    
    temp_ER = temp_EB - temp_EL - outs_left + MoB

    
    
    ER_list.append(temp_ER)
    index.append(i)

exp_runs = pd.DataFrame()
exp_runs['State'] = index
exp_runs['E(B)'] = EB_list
exp_runs['E(L)'] = EL_list
exp_runs['E(R)'] = ER_list
exp_runs = exp_runs.set_index(exp_runs.State)  

display(exp_runs[['E(B)']])

display(exp_runs[['E(L)']])

runs = exp_runs[['E(R)']]

base_dict = {
    0: 'xxx', 
    1: '1xx', 
    2: 'x2x',
    4: 'xx3',
    3: '12x',
    5: '1x3', 
    6: 'x23',
    7: '123'
}

run_matrix = pd.DataFrame()
run_matrix['Bases_Occupied'] = [base_dict[i] for i in base_dict]
for a in range(0, 3):
    if a ==1:
        run_matrix['1_Out'] = np.nan
    else:
        run_matrix['{}_Outs'.format(a)] = np.nan
        
run_matrix = run_matrix.set_index('Bases_Occupied')

for i in runs.index:
    outs = int(i.split('-')[1])
    state = int(i.split('-')[0])
    col = [i for i in run_matrix.columns if str(outs) in i][0]
    run_matrix.at[base_dict[state], 
                  col] = runs.loc[i]['E(R)']

display(run_matrix)

