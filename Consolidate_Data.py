#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 13:23:46 2021

@author: christineanagnos
"""

import pandas as pd
import numpy as np
from datetime import datetime

df_use = pd.read_csv('CleanedData.csv')
df = pd.DataFrame()

startTime = datetime.now()


for ID in df_use.Inning_Pair_ID.unique():
    df_temp = df_use[df_use.Inning_Pair_ID == ID]
    df_t = df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].head(1)
    df_b = df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].head(1)
    
    if len(df_b) >0:
    
    
        t_runs = df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].Total_Runs.max()

        b_runs = df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].Total_Runs.max()


        
        if '2-0' in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].State_Out_Combo.tolist():
            double_binary_t = 1
        if '2-0' not in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].State_Out_Combo.tolist():
            double_binary_t = 0
        if '2-0' in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].State_Out_Combo.tolist():
            double_binary_b = 1
        if '2-0' not in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].State_Out_Combo.tolist():
            double_binary_b = 0
            
        if '1-0' in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].State_Out_Combo.tolist():
            single_binary_t = 1
        if '1-0' not in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].State_Out_Combo.tolist():
            single_binary_t = 0
        if '1-0' in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].State_Out_Combo.tolist():
            single_binary_b = 1
        if '1-0' not in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].State_Out_Combo.tolist():
            single_binary_b = 0
            
        if '3-0' in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].State_Out_Combo.tolist():
            first_second_binary_t = 1
        if '3-0' not in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].State_Out_Combo.tolist():
            first_second_binary_t = 0
        if '3-0' in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].State_Out_Combo.tolist():
            first_second_binary_b = 1
        if '3-0' not in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].State_Out_Combo.tolist():
            first_second_binary_b = 0
            
        if '4-0' in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].State_Out_Combo.tolist():
            triple_binary_t = 1
        if '4-0' not in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].State_Out_Combo.tolist():
            triple_binary_t = 0
        if '4-0' in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].State_Out_Combo.tolist():
            triple_binary_b = 1
        if '4-0' not in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].State_Out_Combo.tolist():
            triple_binary_b = 0
            
        if '5-0' in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].State_Out_Combo.tolist():
            F_T_binary_t = 1
        if '5-0' not in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].State_Out_Combo.tolist():
            F_T_binary_t = 0
        if '5-0' in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].State_Out_Combo.tolist():
            F_T_binary_b = 1
        if '5-0' not in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].State_Out_Combo.tolist():
            F_T_binary_b = 0
            
        if '6-0' in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].State_Out_Combo.tolist():
            S_T_binary_t = 1
        if '6-0' not in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].State_Out_Combo.tolist():
            S_T_binary_t = 0
        if '6-0' in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].State_Out_Combo.tolist():
            S_T_binary_b = 1
        if '6-0' not in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].State_Out_Combo.tolist():
            S_T_binary_b = 0
        
        if '7-0' in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].State_Out_Combo.tolist():
            Loaded_binary_t = 1
        if '7-0' not in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 't' in i])].State_Out_Combo.tolist():
            Loaded_binary_t = 0
        if '7-0' in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].State_Out_Combo.tolist():
            Loaded_binary_b = 1
        if '7-0' not in df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 'b' in i])].State_Out_Combo.tolist():
            Loaded_binary_b = 0

        df_new = pd.DataFrame()

        df_new['Game'] = pd.Series(df_temp.Game.unique()[0])
        df_new['Date'] = pd.Series(df_temp.Date.unique()[0])
        df_new['T_Rank'] = pd.Series(df_temp.T_Rank.unique()[0])
        df_new['B_Rank'] = pd.Series(df_temp.B_Rank.unique()[0])
        df_new['T_WL'] = pd.Series(df_temp.T_WL.unique()[0])
        df_new['B_WL'] = pd.Series(df_temp.B_WL.unique()[0])
        df_new['WL_Diff'] = pd.Series(df_temp.WL_Diff.unique()[0])
        df_new['Rank_Diff'] = pd.Series(df_temp.Rank_Diff.unique()[0])
        df_new['Inning_Pair_ID'] = pd.Series(ID)
        df_new['Inning'] = pd.Series(int(df_t.Inn.tolist()[0][1:]))
        df_new['Score'] = pd.Series(df_t.Score.tolist()[0])
        df_new['T_Mean_BA'] = pd.Series(df_t.Mean_BA.tolist()[0])
        df_new['B_Mean_BA'] = pd.Series(df_b.Mean_BA.tolist()[0])
        df_new['T_Mean_SLG'] = pd.Series(df_t.Mean_SLG.tolist()[0])
        df_new['B_Mean_SLG'] = pd.Series(df_b.Mean_SLG.tolist()[0])
        df_new['T_Mean_OPS'] = pd.Series(df_t.Mean_OPS.tolist()[0])
        df_new['B_Mean_OPS'] = pd.Series(df_b.Mean_OPS.tolist()[0])
        df_new['T_Mean_OPS_Plus'] = pd.Series(df_t.Mean_OPS_Plus.tolist()[0])
        df_new['B_Mean_OPS_Plus'] = pd.Series(df_b.Mean_OPS_Plus.tolist()[0])
        df_new['T_WHIP'] = pd.Series(df_t.Mean_WHIP.tolist()[0])
        df_new['B_WHIP'] = pd.Series(df_b.Mean_WHIP.tolist()[0])
        df_new['T_ERA'] = pd.Series(df_t.Mean_ERA.tolist()[0])
        df_new['B_ERA'] = pd.Series(df_b.Mean_ERA.tolist()[0])
        df_new['T_ERA_Plus'] = pd.Series(df_t.Mean_ERA_Plus.tolist()[0])
        df_new['B_ERA_Plus'] = pd.Series(df_b.Mean_ERA_Plus.tolist()[0])
        
        df_new['T_State_Out_Combos'] = np.nan
        df_new['T_State_Out_Combos'] = df_new['T_State_Out_Combos'].astype(object)
        df_new.at[0, 'T_State_Out_Combos'] = np.array(df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 
                                                                     't' in i])].State_Out_Combo.tolist())
        df_new['B_State_Out_Combos'] = np.nan
        df_new['B_State_Out_Combos'] = df_new['B_State_Out_Combos'].astype(object)
        df_new.at[0, 'B_State_Out_Combos'] = np.array(df_temp[df_temp.Inn.isin([i for i in df_temp.Inn if 
                                                                     'b' in i])].State_Out_Combo.tolist())
        
         
        df_new['T_Runs'] = pd.Series(t_runs)
        df_new['B_Runs'] = pd.Series(b_runs)
        df_new['Run_Diff'] = pd.Series(t_runs - b_runs)

        df_new['Double_Binary_T'] = pd.Series(double_binary_t)
        df_new['Double_Binary_B'] = pd.Series(double_binary_b)
        df_new['Single_Binary_T'] = pd.Series(single_binary_t)
        df_new['Single_Binary_B'] = pd.Series(single_binary_b)
        df_new['F_S_Binary_T'] = pd.Series(first_second_binary_t)
        df_new['F_S_Binary_B'] = pd.Series(first_second_binary_b)
        df_new['Triple_Binary_T'] = pd.Series(triple_binary_t)
        df_new['Triple_Binary_B'] = pd.Series(triple_binary_b)
        df_new['F_T_Binary_T'] = pd.Series(F_T_binary_t)
        df_new['F_T_Binary_B'] = pd.Series(F_T_binary_b)
        df_new['S_T_Binary_B'] = pd.Series(S_T_binary_b)
        df_new['S_T_Binary_T'] = pd.Series(S_T_binary_t)
        df_new['Loaded_Binary_T'] = pd.Series(Loaded_binary_t)
        df_new['Loaded_Binary_B'] = pd.Series(Loaded_binary_b)

        df = pd.concat([df, df_new])
        
    else:
        pass

print(datetime.now() - startTime)

df['Date'] = pd.to_datetime(df.Date)

binary_cond = [
    df.Run_Diff != 0,
    df.Run_Diff == 0
]
binary_vals = [1, 0]
df['Discrepancy'] = np.select(binary_cond, binary_vals)



df.to_csv('Consolidated.csv')