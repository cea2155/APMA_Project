#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:07:27 2021

@author: christineanagnos
"""

import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

startTime = datetime.now()

url_df = pd.read_csv('games.csv')

standings = pd.read_csv('Standings.csv').iloc[:, :2]
batters = pd.read_csv('Batters.csv').iloc[:, :2]
pitchers = pd.read_csv('Pitchers.csv').iloc[:, :2]

# get standings for each year
standings_dict = {}
for year in [i for i in standings['Year'] if pd.isnull(i) == False]:
    url = standings[standings['Year'] == year].reset_index()['Standings'][0]
    df_temp = pd.read_html(url)[0][['Rk', 'Tm', 'W-L%']]
    standings_dict[str(int(year))] = df_temp
    
# get batters for each year
batters_dict = {}
for year in [i for i in batters['Year'] if pd.isnull(i) == False]:
    url = batters[batters['Year'] == year].reset_index()['Batters'][0]
    df_temp = pd.read_html(url)[0][['Name', 'Rk', 'BA', 'SLG', 'OPS', 'OPS+']]
    df_temp['Name'] = [i.replace('\xa0', ' ').strip('*').strip('#').replace('í', 'i').replace('B.J. Upton', 'Melvin Upton Jr.') for i in df_temp.Name]
    batters_dict[str(int(year))] = df_temp

# get pitchers for each year
pitchers_dict = {}
for year in [i for i in pitchers['Year'] if pd.isnull(i) == False]:
    url = pitchers[pitchers['Year'] == year].reset_index()['Pitchers'][0]
    df_temp = pd.read_html(url)[0][['Name', 'Rk', 'WHIP', 'ERA', 'ERA+']]
    df_temp['Name'] = [i.replace('\xa0', ' ').strip('*').strip('#').replace('í', 'i').replace('B.J. Upton', 'Melvin Upton Jr.') for i in df_temp.Name]
    pitchers_dict[str(int(year))] = df_temp
    

df = pd.DataFrame()
inn_id = 0
game_id = 0
inning_pair_id = 0

#parse through dates of games
for column in [i[:10] for i in url_df.columns if '-' in i]:
    date = datetime.strptime(column, '%m-%d-%Y')
    str_year = str(date.year)
    
    # extract standings, batters, and pitchers dataframes based on year
    standings_df = standings_dict[str_year]
    batters_df = batters_dict[str_year]
    pitchers_df = pitchers_dict[str_year]
    
    # parse through tables for individual games
    for url in [i for i in url_df[column] if pd.isnull(i) == False]:
        print(url)
        table_temp = pd.read_html(url)[0]
        
        # get teams playing and compare stats:
        team1_temp = [i for i in table_temp.Inn.unique() if 'Batting' in str(i)][0].split(',')[1].split('Batting')[0].strip(' ')
        team2_temp = [i for i in table_temp.Inn.unique() if 'Batting' in str(i)][0].split(',')[3].split(' ')[1].strip("'")
        team1 = [i for i in standings_df.Tm.unique() if team1_temp in str(i)][0]
        team2 = [i for i in standings_df.Tm.unique() if team2_temp in str(i)][0]
        game = '{}/{}'.format(team1, team2)
        rank_diff = np.abs(standings_df[standings_df.Tm == team1].reset_index().Rk[0]-
                   standings_df[standings_df.Tm == team2].reset_index().Rk[0])
        wl_diff = np.abs(standings_df[standings_df.Tm == team1].reset_index()['W-L%'][0]-
                 standings_df[standings_df.Tm == team2].reset_index()['W-L%'][0])
        
        t_rank = standings_df[standings_df.Tm == team1].reset_index().Rk[0]
        b_rank = standings_df[standings_df.Tm == team2].reset_index().Rk[0]
        t_wl = standings_df[standings_df.Tm == team1].reset_index()['W-L%'][0]
        b_wl = standings_df[standings_df.Tm == team2].reset_index()['W-L%'][0]
        
        

        # get inning pairs (used to compare runs scored later)
        for inn in [a for a in table_temp['Inn'].unique() if 't' in str(a) and len(str(a)) < 5 or
                         'b' in str(a) and len(str(a))<5]:
            
        
            if inn[0] == 't':
                inn_pair = inning_pair_id
                
            if inn[0] == 'b':
                inn_pair = inning_pair_id
                inning_pair_id += 1
                

            
            df_temp = table_temp[table_temp['Inn'] == inn][['Batter', 'Pitcher','Inn','Score', 'Out', 'RoB', 'R/O']]
            df_temp['Game'] = game
            df_temp['Date'] = date
            df_temp['WL_Diff'] = wl_diff
            df_temp['Rank_Diff'] = rank_diff
            df_temp['T_Rank'] = t_rank
            df_temp['B_Rank'] = b_rank
            df_temp['T_WL'] = t_wl
            df_temp['B_WL'] = b_wl
            
            # include data from batter dataframe
            #clean player names:
            df_temp['Batter'] = [i.replace('\xa0', ' ').strip('*').strip('#') for i in df_temp.Batter]
            batters = [i.replace('\xa0', ' ').strip('*').strip('#').replace('í', 'i').replace('B.J. Upton', 'Melvin Upton Jr.') for i in df_temp.Batter]
            ba_lst = []
            slg_lst = []
            ops_lst = []
            ops_plus_lst = []
            for i in batters:
                if len(batters_df.loc[batters_df.Name.str.contains(i, case = True)]) > 0:
                    ba = float(batters_df[batters_df.Name == i].BA.tolist()[0])
                    ba_lst.append(ba)
                    slg = float(batters_df[batters_df.Name == i].SLG.tolist()[0])
                    slg_lst.append(slg)
                    ops = float(batters_df[batters_df.Name == i].OPS.tolist()[0])
                    ops_lst.append(ops)
                    ops_plus = float(batters_df[batters_df.Name == i]['OPS+'].tolist()[0])
                    ops_plus_lst.append(ops_plus)
                    
                #take care of nicknames
                else: 
                    check_names = [a for a in batters_df.Name if 
                                   i.split(' ')[1] in a and a.split(' ')[0][0] == i.split(' ')[0][0]]
                    if len(check_names) == 1:
                        name = check_names[0]
                        ba = float(batters_df[batters_df.Name == name].BA.tolist()[0])
                        ba_lst.append(ba)
                        slg = float(batters_df[batters_df.Name == name].SLG.tolist()[0])
                        slg_lst.append(slg)
                        ops = float(batters_df[batters_df.Name == name].OPS.tolist()[0])
                        ops_lst.append(ops)
                        ops_plus = float(batters_df[batters_df.Name == name]['OPS+'].tolist()[0])
                        ops_plus_lst.append(ops_plus)
                    
                    else:
                        print(check_names)
                        print(i)
                        ba_lst.append(np.nan)
                        slg_lst.append(np.nan)
                        ops_lst.append(np.nan)
                        ops_plus_lst.append(np.nan)

            mean_ba = np.mean(ba_lst)
            mean_slg = np.mean(slg_lst)
            mean_ops = np.mean(ops_lst)
            mean_ops_plus = np.mean(ops_plus_lst)
            
            df_temp['BA'] = ba_lst
            df_temp['SLG'] = slg_lst
            df_temp['OPS'] = ops_lst
            df_temp['OPS_Plus'] = ops_plus_lst
            df_temp['Mean_BA'] = mean_ba
            df_temp['Mean_SLG'] = mean_slg
            df_temp['Mean_OPS'] = mean_ops
            df_temp['Mean_OPS_Plus'] = mean_ops_plus
            
            
            # include data from pitcher dataframe
            #clean player names:
            df_temp['Pitcher'] = [i.replace('\xa0', ' ').strip('*').strip('#').replace('í', 'i').replace('B.J. Upton', 'Melvin Upton Jr.') for i in df_temp.Pitcher]
            pitchers = [i for i in df_temp.Pitcher]
            whip_lst = []
            era_lst = []
            era_plus_lst = []
            for i in pitchers:
                if len(pitchers_df.loc[pitchers_df.Name.str.contains(i, case = True)]) > 0:
                    whip = float(pitchers_df[pitchers_df.Name == i].WHIP.tolist()[0])
                    whip_lst.append(whip)
                    era = float(pitchers_df[pitchers_df.Name == i].ERA.tolist()[0])
                    era_lst.append(era)
                    era_plus = float(pitchers_df[pitchers_df.Name == i]['ERA+'].tolist()[0])
                    era_plus_lst.append(ops_plus)
                    
                #take care of nicknames
                else: 
                    check_names = [a for a in pitchers_df.Name if 
                                   i.split(' ')[1] in a and a.split(' ')[0][0] == i.split(' ')[0][0]]
                    if len(check_names) == 1:
                        name = check_names[0]
                        whip = float(pitchers_df[pitchers_df.Name == name].WHIP.tolist()[0])
                        whip_lst.append(whip)
                        era = float(pitchers_df[pitchers_df.Name == name].ERA.tolist()[0])
                        era_lst.append(era)
                        era_plus = float(pitchers_df[pitchers_df.Name == name]['ERA+'].tolist()[0])
                        era_plus_lst.append(ops_plus)
                    
                    else:
                        print(check_names)
                        print(i)
                        whip_lst.append(np.nan)
                        era_lst.append(np.nan)
                        era_plus_lst.append(np.nan)
                    

            mean_whip = np.mean(whip_lst)
            mean_era = np.mean(era_lst)
            mean_era_plus = np.mean(era_plus_lst)
            
            df_temp['WHIP'] = whip_lst
            df_temp['ERA'] = era_lst
            df_temp['ERA_Plus'] = era_plus_lst
            df_temp['Mean_WHIP'] = mean_whip
            df_temp['Mean_ERA'] = mean_era
            df_temp['Mean_ERA_Plus'] = mean_era_plus
        

            # outs and runs per play
            df_temp['Play_Outs'] = [str(i).count('O') for i in df_temp['R/O']]
            df_temp['Play_Runs'] = [str(i).count('R') for i in df_temp['R/O']]
            
            # separate innings and games
            df_temp['Inning_ID'] = inn_id
            df_temp['Game_ID'] = game_id
            df_temp['Inning_Pair_ID'] = inn_pair
            inn_id += 1
        
            
            # total runs in the inning for each at bat
            total_runs = []
            for i in range(1, len(df_temp.Play_Runs)+1):
                temp_list = df_temp.Play_Runs.tolist()[:i]
                total_runs.append(sum(temp_list))
            df_temp['Total_Runs'] = total_runs
            
            # numerical notation for each combination of runners on base
            cond = [
                df_temp.RoB == '---',
                df_temp.RoB == '1--',
                df_temp.RoB == '-2-',
                df_temp.RoB == '12-',
                df_temp.RoB == '--3',
                df_temp.RoB == '1-3',
                df_temp.RoB == '-23',
                df_temp.RoB == '123',
            ]
            vals = [0,1,2,3,4,5,6,7]
            df_temp['State_Bases'] = np.select(cond, vals)
            
            # include 3 out row (used later in transition matrix)
            df_last = df_temp.iloc[-1:]
            df_last.loc[df_last.index[0], ['Out']] = 3
            df_temp = pd.concat([df_temp, df_last])

            # create column that denotes combination of base state and outs
            df_temp['State_Out_Combo'] = df_temp.State_Bases.astype(
                str) + '-' + df_temp.Out.astype(str)

            # create column that shows transitions between plays
            transition_list = [np.nan]
            for i in range(1, len(df_temp)):
                transition = df_temp.iloc[i-1:i].State_Out_Combo[
                    df_temp.iloc[i-1:i].State_Out_Combo.index[0]] + " to " + df_temp.iloc[
                    i:i+1].State_Out_Combo[
                    df_temp.iloc[i:i+1].State_Out_Combo.index[0]]
                transition_list.append(transition)
            df_temp['Transition'] = transition_list

            # vertically stack dataframes
            df = pd.concat([df, df_temp])
            
            # work thru i's
            last_inn = inn
        
        game_id += 1
        
df.to_csv('CleanedData.csv')

transition_list = [i for i in df.Transition.unique() if i is not np.nan]
total = len([i for i in df.Transition if i is not np.nan])
prob_list = []
for i in transition_list:
    prob_list.append((len(df[df.Transition == i]))/total)
    

transitions = pd.DataFrame()
transitions['Transition'] = transition_list
transitions['Frequency'] = prob_list

transitions.sort_values(by = ['Frequency'], ascending = False).to_csv(
        'Transitions.csv')

print(datetime.now() - startTime)
