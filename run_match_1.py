#!/usr/bin/env python
# coding: utf-8

# In[1]:


import discord
import random
import os
import asyncio

import pandas as pd
import requests


# In[2]:


#########################


# In[3]:


import pandas as pd
import glob
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:

import random
random.seed(42)  


#########################

m = match_number
#m = globals()['i']
for i in range(m, m+1):

    #df_all = df_all
    teams = matches_list.iloc[i][1:].values[0], matches_list.iloc[i][1:].values[1]
    ############
    ground = teams[0]
    toss_winner = random.sample((0,1),1)[0]
    toss_win_team = teams[toss_winner]
    other_team = teams[1-toss_winner]
    decision = random.sample(('Bat','Bowl'),1)[0]

    if decision=='Bat':
        batting_team = team_map(toss_win_team)
        bowling_team = team_map(other_team)
    else:
        bowling_team = team_map(toss_win_team)
        batting_team = team_map(other_team)

    print(f"match {i+1}:: {teams[0]} vs {teams[1]}")
    print(f"toss :: {toss_win_team} wins, and will {decision} first!")

    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################


    ## inning 1

    m_id = 'M00' + str(i+1) if i<9 else 'M0' + str(i+1)
    current_date = matches_list.iloc[i][0]

    match_context = {
        'match_id': m_id,
        'season': 2025,
        'start_date': current_date,
        'venue': ground,  
        'innings': 1,
        'ball': 1
    }

    # Placeholder for output DataFrame
    columns = ['match_id', 'season', 'start_date', 'venue', 'innings', 'ball',
               'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler',
               'runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes',
               'wicket_type', 'player_dismissed','legal_balls_bowled']

    df_simulation = pd.DataFrame(columns=columns)

    batting_order = batting_team.Func_Name.values
    bowling_order = bowling_team[~(bowling_team.bowl.isna())].Func_Name.values

    # Initial striker, non-striker, and bowler
    striker = batting_order[0]
    non_striker = batting_order[1]

    form_factor_bat = form_ba[form_ba.striker.isin(batting_order)].set_index('striker')['form'].to_dict()
    form_factor_bowl = form_bo[form_bo.bowler.isin(bowling_order)].set_index('bowler')['form'].to_dict()

    dismissed_batters = set()  # Set of batters who have already been dismissed

    # Function to retrieve the next batter
    batting_order_iterator = iter(batting_order)  # Iterator over batting order

    # Initialize the first two batters as striker and non-striker
    striker = next(batting_order_iterator)
    non_striker = next(batting_order_iterator)


    class BattingOrder:
        def __init__(self, batting_order):
            self.batting_order_iterator = iter(batting_order)
            self.dismissed_batters = set()

        def next_batsman(self):
            try:
                next_batter = next(self.batting_order_iterator)
                while next_batter in self.dismissed_batters:
                    next_batter = next(self.batting_order_iterator)
            except StopIteration:
                next_batter = None
            return next_batter

    """    
    def next_batsman():
        "Fetch the next available batter from the batting order."
        global batting_order_iterator  # Use the global iterator to keep the sequence

        try:
            next_batter = next(batting_order_iterator)
            while next_batter in dismissed_batters:
                next_batter = next(batting_order_iterator)  # Skip dismissed batters
        except StopIteration:
            next_batter = None  # No more batters available (all-out scenario)

        return next_batter

    """
    # Filter out bowlers and create a list of bowlers with their priorities
    bowlers_df = bowling_team.dropna(subset=['bowl']).reset_index(drop=True)
    bowlers_df['bowl'] = bowlers_df['bowl'].astype(int)  # Ensure 'bowl' column is integer type

    # Initialize tracking for overs bowled and the last over bowled by each bowler
    overs_bowled = {row['Func_Name']: 0 for _, row in bowlers_df.iterrows()}
    last_over_bowled = {row['Func_Name']: -2 for _, row in bowlers_df.iterrows()}  # Start with -2 for all bowlers
    wickets_taken = {row['Func_Name']: 0 for _,row in bowlers_df.iterrows()}
    runs_conceeded_over = {row['Func_Name']: 0 for _,row in bowlers_df.iterrows()}

    ##
    ##
    ##

    # Function to calculate probabilities based on bowler priority and conditions
    def calculate_bowler_probabilities(bowlers_df, overs_bowled, last_over_bowled, over, wickets_taken,runs_conceeded_over):
        probabilities = []

        for _, row in bowlers_df.iterrows():
            name = row['Func_Name']
            rank = row['bowl']

            # Initialize base probability by bowler rank
            prob = 0.6 if rank <= 5 else 0.05

            # Add preference for higher-priority bowlers in initial overs
            if rank <= 2 and over < 5:
                prob += 0.2

            # Add preference for higher-priority bowlers in final overs
            if rank <= 3 and over > 15 and overs_bowled[name] < 4:
                prob += min(1.25, -0.2+1/prob)

            if ((wickets_taken[name] >=1) or (runs_conceeded_over[name]/(overs_bowled[name] if overs_bowled[name]>=1 else 1) <=8.5)) and last_over_bowled[name] == over - 2:
                prob *= min(1.4, -0.2+1/prob)

            # Slightly increase probability if bowler rested last over (bowled two overs ago)
            elif last_over_bowled[name] <= over - 3:
                prob *= min(1.15, -0.2+1/prob)

            # Reduce probability for lower-priority bowlers who have already bowled 2 or more overs
            if overs_bowled[name] >= 3 and rank >= 4:
                prob *= min(0.5, -0.2+1/prob)

            if last_over_bowled[name] == over - 1:
                prob = 0  # Ensure no consecutive overs

            if overs_bowled[name] == 4:
                prob = 0  # Ensure not more than 4 overs

            probabilities.append(prob)

        # Normalize probabilities to sum up to 1
        total_prob = sum(probabilities)
        return [p / total_prob for p in probabilities] if total_prob > 0 else [1/len(probabilities)] * len(probabilities)

    # Initialize counters and simulation settings
    legal_balls = 0
    legal_balls_last = 0
    all_balls = 0
    wickets_down = 0
    runs_conceeded_o = 0


    # Initialize bowler for the first over
    over = 1
    bowler_probabilities = calculate_bowler_probabilities(bowlers_df, overs_bowled, last_over_bowled, over, wickets_taken,overs_bowled)
    bowler = np.random.choice(bowlers_df['Func_Name'], p=bowler_probabilities)

    for over in range(1, 21):
        for ball in range(1, 10):  # Accounting for max 10 balls in case of extras in each over
            if legal_balls == 120 or wickets_down==10:
                break  # End loop when 120 legal balls are bowled (20 overs)
            ####


            h2h_bat = h2h[(h2h.striker==striker)&(h2h.bowler==bowler)].h2h_factor_bat.values
            h2h_bowl = h2h[(h2h.striker==striker)&(h2h.bowler==bowler)].h2h_factor_bowl.values

            current_ball = (over - 1) * 6 + ball  # Track overall ball count
            all_balls += 1

            # Determine phase of play: Powerplay, Middle, Death
            phase = 'pp' if over <= 6 else 'middle' if over <= 15 else 'death'

            # Retrieve probabilities based on the current ground, batter, and bowler stats
            ground_probs = df_g[(df_g.venue == match_context['venue']) & 
                                (df_g.innings == match_context['innings']) & 
                                (df_g.phase == phase)].iloc[0]

            filtered_batter_probs = df_ba[(df_ba.striker == striker) & 
                                  (df_ba.innings == match_context['innings']) & 
                                  (df_ba.phase == phase)]

            if not filtered_batter_probs.empty:
                batter_probs = filtered_batter_probs.iloc[0]
            else:
                batter_probs = pd.Series({col: 1 for col in df_ba.columns})  # Default values for all columns

            # Bowler probabilities
            filtered_bowler_probs = df_bo[(df_bo.bowler == bowler) & 
                                          (df_bo.innings == match_context['innings']) & 
                                          (df_bo.phase == phase)]

            if not filtered_bowler_probs.empty:
                bowler_probs = filtered_bowler_probs.iloc[0]
            else:
                bowler_probs = pd.Series({col: 1 for col in df_bo.columns}) 



            # Adjust probabilities using form and H2H factors
            final_probs = {
                'one': max(0.01, np.random.normal(0, 0.015) + (ground_probs['one_prob'] + batter_probs['one_prob'] + bowler_probs['one_prob']) * form_factor_bat.get(striker, 1) / 3),
                'two': max(0.01,np.random.normal(0, 0.015) + (ground_probs['two_prob'] + batter_probs['two_prob'] + bowler_probs['two_prob']) * form_factor_bat.get(striker, 1) / 3),
                'three': max(0.01,np.random.normal(0,0.015) + (ground_probs['three_prob'] + batter_probs['three_prob'] + bowler_probs['three_prob']) * form_factor_bat.get(striker, 1) / 3),
                'four': max(0.01,np.random.normal(0, 0.015) + (ground_probs['four_prob'] + batter_probs['four_prob'] + bowler_probs['four_prob']) * form_factor_bat.get(striker, 1) / 3),
                'six': max(0.01,np.random.normal(0, 0.015) + (ground_probs['six_prob'] + batter_probs['six_prob'] + bowler_probs['six_prob']) * form_factor_bat.get(striker, 1) / 3),
                'dot': max(0.01,np.random.normal(0,0.015) + (ground_probs['dot_prob'] + batter_probs['dot_prob'] + bowler_probs['dot_prob']) * (1 / form_factor_bat.get(striker, 1)) / 3),
                'wicket': max(0.01,np.random.normal(0, 0.015) + (ground_probs['wkt_prob'] + batter_probs['out_prob'] + bowler_probs['wkt_prob']) * form_factor_bowl.get(bowler, 1) * (h2h_bowl[0] if len(h2h_bowl)>0 else 1) / 3),
                '0+runout': (0.005)/3,
                '1+runout': (0.005)/3,
                '2+runout': (0.005)/3,
                '3+runout': 0.0005,
                'wide': max(0.01,np.random.normal(0, 0.02) + ground_probs['wide_prob']),
                'noball': ground_probs.get('no_prob', 0.01),
                'bye': ground_probs.get('bye_prob', 0.01),
                'legbye': ground_probs.get('legbye_prob', 0.01)
            }

            # Normalize probabilities so they sum up to 1
            total = sum(final_probs.values())
            final_probs = {k: v / total for k, v in final_probs.items()}

            # Simulate outcome of the ball
            event = np.random.choice(list(final_probs.keys()), p=np.ravel(list(final_probs.values())))

            # Update legal ball count for valid deliveries
            legal_balls += 1 if event not in ['wide', 'noball'] else 0
            wickets_down += 1 if event in ['wicket', '0+runout','1+runout','2+runout','3+runout'] else 0

            # Capture details of this ball
            row = match_context.copy()
            row.update({
                'innings': match_context['innings'],
                'ball': all_balls, 
                'striker': striker,
                'non_striker': non_striker,
                'bowler': bowler,
                'runs_off_bat': 1 if event in ['one','1+runout'] else 2 if event in ['2+runout','two'] else 3 if event in ['3+runout','three'] else 4 if event == 'four' else 6 if event == 'six' else 0,
                'extras': 1 if event in ['wide', 'noball', 'bye', 'legbye'] else 0,
                'wides': 1 if event == 'wide' else 0,
                'noballs': 1 if event == 'noball' else 0,
                'byes': 1 if event == 'bye' else 0,
                'legbyes': 1 if event == 'legbye' else 0,
                'wicket_type': 'out' if event == 'wicket' else 'runout' if event in ['0+runout','1+runout','2+runout','3+runout'] else None,
                'player_dismissed': striker if event in ['wicket','0+runout','2+runout'] else non_striker if event in ['1+runout','3+runout'] else None,
                'legal_balls_bowled': legal_balls
            })

            # Append data for each ball to simulation DataFrame
            df_simulation = pd.concat([df_simulation, pd.DataFrame([row])], ignore_index=True)

            print(f"{bowler} to {striker} : {event}")


            """def handle_event(event, df_simulation, striker, non_striker):
                if event in ['wicket', '0+runout', '2+runout']:
                    striker_runs = df_simulation[df_simulation.striker == striker].runs_off_bat.sum()
                    print(f"Batter out: {striker}, for {striker_runs}")
                    striker = batting_order.next_batsman()
                elif event in ['1+runout', '3+runout']:
                    non_striker_runs = df_simulation[df_simulation.striker == non_striker].runs_off_bat.sum()
                    print(f"Batter out: {non_striker}, for {non_striker_runs}")
                    non_striker = batting_order.next_batsman()
                elif event in ['one', 'three', 'bye', 'legbye']:
                    striker, non_striker = non_striker, striker  # Swap striker and non-striker

                return striker, non_striker
            """
            #striker, non_striker = handle_event(event, df_simulation, striker, non_striker)

            # Switch batters on odd runs or for a wicket
            if event in ['wicket','0+runout','2+runout']:
                striker_runs = df_simulation[df_simulation.striker==striker].runs_off_bat.sum()
                print(f"batter out: {striker}, for {striker_runs}")
                striker = next_batsman() #############
            elif event in ['1+runout','3+runout']:
                non_striker_runs = df_simulation[df_simulation.striker==non_striker].runs_off_bat.sum()
                print(f"batter out: {non_striker}, for {non_striker_runs}")
                non_striker = next_batsman()
            elif event in ['one', 'three','bye','legbye']:
                striker, non_striker = non_striker, striker  # Swap striker and non-striker


            runs_conceeded_o += row['runs_off_bat']+row['extras']
            if event == 'wicket':
                wickets_taken[bowler] += 1
            # Change bowler every 6 legal balls (end of an over)
            if (legal_balls % 6 == 0)&(legal_balls!=legal_balls_last):
                striker, non_striker = non_striker, striker #over ends
                overs_bowled[bowler] += 1
                runs_conceeded_over[bowler] += runs_conceeded_o
                ###
                last_over_bowled[bowler] = over  # Update last over bowled for current bowler
                bowler_probabilities = calculate_bowler_probabilities(bowlers_df, overs_bowled, last_over_bowled, over + 1, wickets_taken,runs_conceeded_over)
                bowler = np.random.choice(bowlers_df['Func_Name'], p=bowler_probabilities)

                score = df_simulation['runs_off_bat'].sum()+df_simulation['extras'].sum()
                print(f"end of over {legal_balls//6}; score :: {score} for {wickets_down}")
                print('---'*5)

                runs_conceeded_o = 0
            legal_balls_last += 1 if event not in ['wide', 'noball'] else 0


    df_mod = func_1(df_simulation)

    df_mod_1 = func_2(df_mod)

    bat_1st = (other_team if decision=='Bowl' else toss_win_team)
    bowl_1st = (toss_win_team if decision=='Bowl' else other_team)

    df_mod_1['batting_team'] = bat_1st
    df_mod_1['bowling_team'] = bowl_1st

    columns = ['innings','striker','non_striker','bowler','runs_off_bat','extras','wicket_type','player_dismissed',
              'legal_balls_bowled','runs_scored','bowler_wicket','run_rate','last_fow','reqd_run_rate']
    #df_mod_1#[columns]

    print(f"total: {df_mod_1['runs_scored'].max()}, wickets: {df_mod_1['wickets_down'].max()}")

    print("the chase is underway!!!!!!!!!!!!")

    ## inning 2

    # Setup initial game context
    match_context = {
        'match_id': m_id,
        'season': 2025,
        'start_date': current_date,
        'venue': ground,  # Ground name as provided
        'innings': 2,
        'ball': 1,

    }

    # Placeholder for output DataFrame
    columns = ['match_id', 'season', 'start_date', 'venue', 'innings', 'ball',
               'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler',
               'runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes',
               'wicket_type', 'player_dismissed','legal_balls_bowled']

    df_simulation_2 = pd.DataFrame(columns=columns)

    ##
    batting_team, bowling_team = bowling_team, batting_team
    ##

    batting_order = batting_team.Func_Name.values
    bowling_order = bowling_team[~(bowling_team.bowl.isna())].Func_Name.values

    # Initial striker, non-striker, and bowler
    striker = batting_order[0]
    non_striker = batting_order[1]

    form_factor_bat = form_ba[form_ba.striker.isin(batting_order)].set_index('striker')['form'].to_dict()
    form_factor_bowl = form_bo[form_bo.bowler.isin(bowling_order)].set_index('bowler')['form'].to_dict()


    #h2h_bat = h2h[(h2h.striker==striker)&(h2h.bowler==bowler)].h2h_factor_bat.values
    #h2h_bowl = h2h[(h2h.striker==striker)&(h2h.bowler==bowler)].h2h_factor_bowl.values

    dismissed_batters = set()  # Set of batters who have already been dismissed

    # Function to retrieve the next batter
    batting_order_iterator = iter(batting_order)  # Iterator over batting order

    # Initialize the first two batters as striker and non-striker
    striker = next(batting_order_iterator)
    non_striker = next(batting_order_iterator)

    def next_batsman():
        """Fetch the next available batter from the batting order."""
        global batting_order_iterator  # Use the global iterator to keep the sequence

        try:
            next_batter = next(batting_order_iterator)
            while next_batter in dismissed_batters:
                next_batter = next(batting_order_iterator)  # Skip dismissed batters
        except StopIteration:
            next_batter = None  # No more batters available (all-out scenario)

        return next_batter

    # Filter out bowlers and create a list of bowlers with their priorities
    bowlers_df = bowling_team.dropna(subset=['bowl']).reset_index(drop=True)
    bowlers_df['bowl'] = bowlers_df['bowl'].astype(int)  # Ensure 'bowl' column is integer type

    # Initialize tracking for overs bowled and the last over bowled by each bowler
    overs_bowled = {row['Func_Name']: 0 for _, row in bowlers_df.iterrows()}
    last_over_bowled = {row['Func_Name']: -2 for _, row in bowlers_df.iterrows()}  # Start with -2 for all bowlers
    wickets_taken = {row['Func_Name']: 0 for _,row in bowlers_df.iterrows()}
    runs_conceeded_over = {row['Func_Name']: 0 for _,row in bowlers_df.iterrows()}


    legal_balls = 0
    legal_balls_last = 0
    all_balls = 0
    wickets_down = 0
    runs_scored = 0
    runs_conceeded_o = 0


    target = df_simulation['runs_off_bat'].sum()+df_simulation['extras'].sum()+1
    reqd_run_rate = 6*target/120


    # Initialize bowler for the first over
    over = 1
    bowler_probabilities = calculate_bowler_probabilities(bowlers_df, overs_bowled, last_over_bowled, over, wickets_taken, runs_conceeded_over)
    bowler = np.random.choice(bowlers_df['Func_Name'], p=bowler_probabilities)

    for over in range(1, 21):
        for ball in range(1, 10):  # Accounting for max 10 balls in case of extras in each over

            legal_balls_remaining = 120 - legal_balls

            #runs_remaining = target - runs_scored

            if (legal_balls == 120) or (wickets_down==10) or (runs_scored>=target):
                break  

            h2h_bat = h2h[(h2h.striker==striker)&(h2h.bowler==bowler)].h2h_factor_bat.values
            h2h_bowl = h2h[(h2h.striker==striker)&(h2h.bowler==bowler)].h2h_factor_bowl.values

            current_ball = (over - 1) * 6 + ball  # Track overall ball count
            all_balls += 1

            # Determine phase of play: Powerplay, Middle, Death
            phase = 'pp' if over <= 6 else 'middle' if over <= 15 else 'death'

            bowl_nrr_phase = 'easy' if reqd_run_rate >=10 else 'crucial' if reqd_run_rate<=8 else 'moderate'
            bat_nrr_phase = 'crucial' if reqd_run_rate >=10 else 'easy' if reqd_run_rate<=8 else 'moderate'

            bowl_wkt_phase = 'easy' if wickets_down >=7 else 'crucial' if wickets_down<=3 else 'medium'
            bat_wkt_phase = 'tough' if wickets_down >=7 else 'easy' if wickets_down<=3 else 'medium'


            # Retrieve probabilities based on the current ground, batter, and bowler stats
            ground_probs = df_g[(df_g.venue == match_context['venue']) & 
                                (df_g.innings == match_context['innings']) & 
                                (df_g.phase == phase)].iloc[0]

            filtered_batter_probs = df_ba[(df_ba.striker == striker) & 
                                  (df_ba.innings == match_context['innings']) & 
                                  (df_ba.phase == phase)&
                                  (df_ba.wkt_phase == bat_wkt_phase)&
                                  (df_ba.nrr_phase == bat_nrr_phase)]

            if not filtered_batter_probs.empty:
                batter_probs = filtered_batter_probs.iloc[0]
            else:
                batter_probs = pd.Series({col: 1 for col in df_ba.columns})  # Default values for all columns

            # Bowler probabilities
            filtered_bowler_probs = df_bo[(df_bo.bowler == bowler) & 
                                          (df_bo.innings == match_context['innings']) & 
                                          (df_bo.phase == phase)&
                                          (df_bo.wkt_phase == bowl_wkt_phase)&
                                          (df_bo.nrr_phase == bowl_nrr_phase)]

            if not filtered_bowler_probs.empty:
                bowler_probs = filtered_bowler_probs.iloc[0]
            else:
                bowler_probs = pd.Series({col: 1 for col in df_bo.columns}) 



            # Adjust probabilities using form and H2H factors
            final_probs = {
                'one': max(0.01, np.random.normal(0, 0.015) + (ground_probs['one_prob'] + batter_probs['one_prob'] + bowler_probs['one_prob']) * form_factor_bat.get(striker, 1) / 3),
                'two': max(0.01,np.random.normal(0, 0.015) + (ground_probs['two_prob'] + batter_probs['two_prob'] + bowler_probs['two_prob']) * form_factor_bat.get(striker, 1) / 3),
                'three': max(0.01,np.random.normal(0,0.015) + (ground_probs['three_prob'] + batter_probs['three_prob'] + bowler_probs['three_prob']) * form_factor_bat.get(striker, 1) / 3),
                'four': max(0.01,np.random.normal(0, 0.015) + (ground_probs['four_prob'] + batter_probs['four_prob'] + bowler_probs['four_prob']) * form_factor_bat.get(striker, 1) / 3),
                'six': max(0.01,np.random.normal(0, 0.015) + (ground_probs['six_prob'] + batter_probs['six_prob'] + bowler_probs['six_prob']) * form_factor_bat.get(striker, 1) / 3),
                'dot': max(0.01,np.random.normal(0,0.015) + (ground_probs['dot_prob'] + batter_probs['dot_prob'] + bowler_probs['dot_prob']) * (1 / form_factor_bat.get(striker, 1)) / 3),
                'wicket': max(0.01,np.random.normal(0, 0.015) + (ground_probs['wkt_prob'] + batter_probs['out_prob'] + bowler_probs['wkt_prob']) * form_factor_bowl.get(bowler, 1) * (h2h_bowl[0] if len(h2h_bowl)>0 else 1) / 3),
                '0+runout': (0.005)/3,
                '1+runout': (0.005)/3,
                '2+runout': (0.005)/3,
                '3+runout': 0.0005,
                'wide': max(0.01,np.random.normal(0, 0.02) + ground_probs['wide_prob']),
                'noball': ground_probs.get('no_prob', 0.01),
                'bye': ground_probs.get('bye_prob', 0.01),
                'legbye': ground_probs.get('legbye_prob', 0.01)
            }

            # Normalize probabilities so they sum up to 1
            total = sum(final_probs.values())
            final_probs = {k: v / total for k, v in final_probs.items()}

            # Simulate outcome of the ball
            event = np.random.choice(list(final_probs.keys()), p=np.ravel(list(final_probs.values())))

            # Update legal ball count for valid deliveries
            legal_balls += 1 if event not in ['wide', 'noball'] else 0
            wickets_down += 1 if event in ['wicket', '0+runout','1+runout','2+runout','3+runout'] else 0
            legal_balls_remaining = 120 - legal_balls
            #runs_remaining = target - runs_scored

            # Capture details of this ball
            row = match_context.copy()
            row.update({
                'innings': match_context['innings'],
                'ball': all_balls, 
                'striker': striker,
                'non_striker': non_striker,
                'bowler': bowler,
                'runs_off_bat': 1 if event in ['one','1+runout'] else 2 if event in ['2+runout','two'] else 3 if event in ['3+runout','three'] else 4 if event == 'four' else 6 if event == 'six' else 0,
                'extras': 1 if event in ['wide', 'noball', 'bye', 'legbye'] else 0,
                'wides': 1 if event == 'wide' else 0,
                'noballs': 1 if event == 'noball' else 0,
                'byes': 1 if event == 'bye' else 0,
                'legbyes': 1 if event == 'legbye' else 0,
                'wicket_type': 'out' if event == 'wicket' else 'runout' if event in ['0+runout','1+runout','2+runout','3+runout'] else None,
                'player_dismissed': striker if event in ['wicket','0+runout','2+runout'] else non_striker if event in ['1+runout','3+runout'] else None,
                'legal_balls_bowled': legal_balls,
                'legal_balls_remaining': legal_balls_remaining,
                'target': target
            })

            # Append data for each ball to simulation DataFrame
            df_simulation_2 = pd.concat([df_simulation_2, pd.DataFrame([row])], ignore_index=True)

            runs_ = row['runs_off_bat']+row['extras']
            runs_scored = runs_scored+ runs_

            reqd_run_rate = ((6 * (target-runs_scored) / legal_balls_remaining)                                                if legal_balls_remaining  > 0 else 6 + (target-runs_scored)) 


            print(f"{bowler} to {striker} : {event}, score: {runs_scored}")

            # Switch batters on odd runs or for a wicket
            if event in ['wicket','0+runout','2+runout']:
                striker_runs = df_simulation_2[df_simulation_2.striker==striker].runs_off_bat.sum()
                print(f"batter out: {striker}, for {striker_runs}")
                striker = next_batsman() 
            elif event in ['1+runout','3+runout']:
                non_striker_runs = df_simulation_2[df_simulation_2.striker==non_striker].runs_off_bat.sum()
                print(f"batter out: {non_striker}, for {non_striker_runs}")
                non_striker = next_batsman()
            elif event in ['one', 'three']:
                striker, non_striker = non_striker, striker  # Swap striker and non-striker


            if event == 'wicket':
                wickets_taken[bowler] += 1
            # Change bowler every 6 legal balls (end of an over)
            if (legal_balls % 6 == 0)&(legal_balls!=legal_balls_last):
                striker, non_striker = non_striker, striker #over ends
                overs_bowled[bowler] += 1
                runs_conceeded_over[bowler] += runs_conceeded_o

                last_over_bowled[bowler] = over  # Update last over bowled for current bowler
                bowler_probabilities = calculate_bowler_probabilities(bowlers_df, overs_bowled, last_over_bowled, over + 1, wickets_taken, runs_conceeded_over)
                bowler = np.random.choice(bowlers_df['Func_Name'], p=bowler_probabilities)

                print(f"end of over {legal_balls//6}; score :: {runs_scored} for {wickets_down}")
                print('---'*5)
                runs_conceeded_o = 0
            legal_balls_last += 1 if event not in ['wide', 'noball'] else 0
            #runs_remaining = target - runs_scored

    df_mod = func_1(df_simulation_2)

    df_mod_2 = func_2(df_mod)

    df_mod_2['runs_remaining'] = df_mod_2['target']-df_mod_2['runs_scored']
    df_mod_2['reqd_run_rate'] = np.where(
        df_mod_2['legal_balls_remaining'] > 0,
        (6 * (df_mod_2['target'] - df_mod_2['runs_scored']) / df_mod_2['legal_balls_remaining']),
        6 + (df_mod_2['target'] - df_mod_2['runs_scored'])
    )

    bat_2nd = (other_team if decision=='Bat' else toss_win_team)
    bowl_2nd = (toss_win_team if decision=='Bat' else other_team)

    df_mod_2['batting_team'] = bat_2nd
    df_mod_2['bowling_team'] = bowl_2nd

    columns = ['innings','striker','non_striker','bowler','runs_off_bat','extras','wicket_type','player_dismissed',
              'legal_balls_bowled','bowler_wicket','run_rate','last_fow','reqd_run_rate']
    #df_mod_2#[columns]

    print(f"total: {df_mod_2['runs_scored'].max()}, wickets: {df_mod_2['wickets_down'].max()}")

    df_mod_2[df_mod_2.isWicket>0]

    ##bowling stats

    df_mod_1['runs_conceeded'] = df_mod_1['runs_off_bat']+df_mod_1['wides']+df_mod_1['noballs']

    df_mod_1['isDotforbowler'] = np.where((df_mod_1['runs_conceeded']==0)&(df_mod_1['islegal']==1), 1, 0)

    df_mod_2['runs_conceeded'] = df_mod_2['runs_off_bat']+df_mod_2['wides']+df_mod_2['noballs']

    df_mod_2['isDotforbowler'] = np.where((df_mod_2['runs_conceeded']==0)&(df_mod_2['islegal']==1), 1, 0)

    df_mod = pd.concat([df_mod_1,df_mod_2],axis=0).reset_index(drop=True)

    inn1_score = df_mod[df_mod.innings==1].runs_scored.max()
    inn2_score = df_mod[df_mod.innings==2].runs_scored.max()

    if inn1_score>inn2_score:
        print(f"{bowl_2nd} wins!")
    elif inn1_score<inn2_score:
        print(f"{bat_2nd} wins!")
    else:
        print("it's a tie!!!!!")

    #################

    df_all = df_all.append(df_mod).reset_index(drop=True)


    ##########################################
    t2 = time.time()

########################################################################################
########################################################################################

########################################################################################
########################################################################################

########################################################################################
########################################################################################

########################################################################################
########################################################################################
## bowler stats

bowler_stats = df_all.groupby(['bowler','bowling_team']).agg(   ##,'innings'
    num_innings = ('match_id','nunique'),
    runs = ('runs_conceeded','sum'),
    balls = ('islegal' ,'sum'),
    wkts = ('isBowlerWicket','sum'),
    fours = ('isFour', 'sum'),
    sixes = ('isSix','sum'),
    dots = ('isDotforbowler','sum'),
    
    ones = ('isOne','sum'),
    twos = ('isTwo','sum'),
    threes = ('isThree','sum'),
    wides = ('wides','sum'),
    noballs = ('noballs','sum')
        
    
).reset_index()

bowler_stats['economy'] = 6*bowler_stats['runs']/bowler_stats['balls']
bowler_stats['strike_rate'] = bowler_stats['balls']/bowler_stats['wkts']
bowler_stats['bpb'] = bowler_stats['balls']/(bowler_stats['fours']+bowler_stats['sixes'])
bowler_stats['dot_%'] = 100*bowler_stats['dots']/bowler_stats['balls']

# Sort DataFrame based on the custom order
bowler_stats = bowler_stats.sort_values(['wkts','economy'], ascending=[False, True]).reset_index(drop=True)

#batting stats

batter_stats = df_all.groupby(['striker','batting_team']).agg(  ##,'innings'
    num_innings = ('match_id','nunique'),
    runs = ('total_runs','sum'),
    balls = ('is_faced_by_batter' ,'sum'),
    outs = ('is_striker_Out','sum'),
    fours = ('isFour', 'sum'),
    sixes = ('isSix','sum'),
    dots = ('isDotforBatter','sum'),
    
    ones = ('isOne','sum'),
    twos = ('isTwo','sum'),
    threes = ('isThree','sum')
    
    
    
    
).reset_index()

batter_stats['strike_rate'] = 100*batter_stats['runs']/batter_stats['balls']
batter_stats['balls_per_dismissal'] = batter_stats['balls']/batter_stats['outs']
batter_stats['bpb'] = batter_stats['balls']/(batter_stats['fours']+batter_stats['sixes'])
batter_stats['dot_%'] = 100*batter_stats['dots']/batter_stats['balls']

batter_stats = batter_stats.sort_values(['runs','strike_rate'], ascending=[False,False]).reset_index(drop=True)

df_match_info = pd.DataFrame()

df_match_info['match_id'] = df_all.match_id.unique()
df_match_info['team_1'] = ''
df_match_info['team_2'] = ''
df_match_info['team_1_total'] = 0
df_match_info['team_1_balls'] = 0
df_match_info['team_2_total'] = 0
df_match_info['team_2_balls'] = 0
df_match_info['winner'] = ''


for index,row in df_match_info.iterrows():
    subset_df = df_all[df_all.match_id==row['match_id']][df_all.innings==1]
    t1 = subset_df['batting_team'].unique()[0]
    t1_runs = subset_df['runs_scored'].max()
    t1_balls = subset_df['legal_balls_bowled'].max()
    team_1_total = t1_runs
    team_1_balls = t1_balls
    
    subset_df = df_all[df_all.match_id==row['match_id']][df_all.innings==2]
    t2 = subset_df['batting_team'].unique()[0]
    t2_runs = subset_df['runs_scored'].max()
    t2_balls = subset_df['legal_balls_bowled'].max()
    team_2_total = t2_runs
    team_2_balls = t2_balls
    
    winner = t1 if team_1_total>team_2_total else t2 if team_1_total<team_2_total else 'TIE'
    
    df_match_info.at[index, 'team_1'] = t1
    df_match_info.at[index, 'team_2'] = t2
    df_match_info.at[index, 'team_1_total'] = team_1_total
    df_match_info.at[index, 'team_1_balls'] = team_1_balls
    df_match_info.at[index, 'team_2_total'] = team_2_total
    df_match_info.at[index, 'team_2_balls'] = team_2_balls
    df_match_info.at[index, 'winner'] = winner
    


pts_table = pd.DataFrame(df_match_info.groupby('winner')['team_2_balls'].count()*2).reset_index()

pts_table.columns = ['team','points']


CSK_runs_for = []
CSK_balls_for = []
DC_runs_for = []
DC_balls_for = []
GT_runs_for = []
GT_balls_for = []
KKR_runs_for = []
KKR_balls_for = []
LSG_runs_for = []
LSG_balls_for = []
MI_runs_for = []
MI_balls_for = []
PBKS_runs_for = []
PBKS_balls_for = []
RCB_runs_for = []
RCB_balls_for = []
RR_runs_for = []
RR_balls_for = []
SRH_runs_for = []
SRH_balls_for = []

CSK_runs_against = []
CSK_balls_against = []
DC_runs_against = []
DC_balls_against = []
GT_runs_against = []
GT_balls_against = []
KKR_runs_against = []
KKR_balls_against = []
LSG_runs_against = []
LSG_balls_against = []
MI_runs_against = []
MI_balls_against = []
PBKS_runs_against = []
PBKS_balls_against = []
RCB_runs_against = []
RCB_balls_against = []
RR_runs_against = []
RR_balls_against = []
SRH_runs_against = []
SRH_balls_against = []

for m in df_all.match_id.unique():
    m_df = df_all[df_all.match_id==m].reset_index(drop=True)
    
    m_df_1 = m_df[m_df.innings==1]
    runs_1 = m_df_1.runs_scored.max()
    balls_1 = m_df_1.legal_balls_bowled.max()
    balls_1 = max(120,balls_1)
    
    m_df_2 = m_df[m_df.innings==2]
    runs_2 = m_df_2.runs_scored.max()
    balls_2 = m_df_2.legal_balls_bowled.max()
    wkts_2 = m_df_2.wickets_down.max()
    balls_2 = 120 if wkts_2==10 else balls_2
    
    
        

    if 'CSK' in m_df_1.batting_team.unique():
        CSK_runs_for.append(runs_1)
        CSK_balls_for.append(balls_1)
    if 'DC' in m_df_1.batting_team.unique():
        DC_runs_for.append(runs_1)
        DC_balls_for.append(balls_1)
    if 'GT' in m_df_1.batting_team.unique():
        GT_runs_for.append(runs_1)
        GT_balls_for.append(balls_1)
    if 'KKR' in m_df_1.batting_team.unique():
        KKR_runs_for.append(runs_1)
        KKR_balls_for.append(balls_1)
    if 'LSG' in m_df_1.batting_team.unique():
        LSG_runs_for.append(runs_1)
        LSG_balls_for.append(balls_1)
    if 'MI' in m_df_1.batting_team.unique():
        MI_runs_for.append(runs_1)
        MI_balls_for.append(balls_1)
    if 'PBKS' in m_df_1.batting_team.unique():
        PBKS_runs_for.append(runs_1)
        PBKS_balls_for.append(balls_1)
    if 'RCB' in m_df_1.batting_team.unique():
        RCB_runs_for.append(runs_1)
        RCB_balls_for.append(balls_1)
    if 'RR' in m_df_1.batting_team.unique():
        RR_runs_for.append(runs_1)
        RR_balls_for.append(balls_1)
    if 'SRH' in m_df_1.batting_team.unique():
        SRH_runs_for.append(runs_1)
        SRH_balls_for.append(balls_1)


    if 'CSK' in m_df_2.batting_team.unique():
        CSK_runs_for.append(runs_2)
        CSK_balls_for.append(balls_2)
    if 'DC' in m_df_2.batting_team.unique():
        DC_runs_for.append(runs_2)
        DC_balls_for.append(balls_2)
    if 'GT' in m_df_2.batting_team.unique():
        GT_runs_for.append(runs_2)
        GT_balls_for.append(balls_2)
    if 'KKR' in m_df_2.batting_team.unique():
        KKR_runs_for.append(runs_2)
        KKR_balls_for.append(balls_2)
    if 'LSG' in m_df_2.batting_team.unique():
        LSG_runs_for.append(runs_2)
        LSG_balls_for.append(balls_2)
    if 'MI' in m_df_2.batting_team.unique():
        MI_runs_for.append(runs_2)
        MI_balls_for.append(balls_2)
    if 'PBKS' in m_df_2.batting_team.unique():
        PBKS_runs_for.append(runs_2)
        PBKS_balls_for.append(balls_2)
    if 'RCB' in m_df_2.batting_team.unique():
        RCB_runs_for.append(runs_2)
        RCB_balls_for.append(balls_2)
    if 'RR' in m_df_2.batting_team.unique():
        RR_runs_for.append(runs_2)
        RR_balls_for.append(balls_2)
    if 'SRH' in m_df_2.batting_team.unique():
        SRH_runs_for.append(runs_2)
        SRH_balls_for.append(balls_2)

    if 'CSK' in m_df_1.bowling_team.unique():
        CSK_runs_against.append(runs_1)
        CSK_balls_against.append(balls_1)
    if 'DC' in m_df_1.bowling_team.unique():
        DC_runs_against.append(runs_1)
        DC_balls_against.append(balls_1)
    if 'GT' in m_df_1.bowling_team.unique():
        GT_runs_against.append(runs_1)
        GT_balls_against.append(balls_1)
    if 'KKR' in m_df_1.bowling_team.unique():
        KKR_runs_against.append(runs_1)
        KKR_balls_against.append(balls_1)
    if 'LSG' in m_df_1.bowling_team.unique():
        LSG_runs_against.append(runs_1)
        LSG_balls_against.append(balls_1)
    if 'MI' in m_df_1.bowling_team.unique():
        MI_runs_against.append(runs_1)
        MI_balls_against.append(balls_1)
    if 'PBKS' in m_df_1.bowling_team.unique():
        PBKS_runs_against.append(runs_1)
        PBKS_balls_against.append(balls_1)
    if 'RCB' in m_df_1.bowling_team.unique():
        RCB_runs_against.append(runs_1)
        RCB_balls_against.append(balls_1)
    if 'RR' in m_df_1.bowling_team.unique():
        RR_runs_against.append(runs_1)
        RR_balls_against.append(balls_1)
    if 'SRH' in m_df_1.bowling_team.unique():
        SRH_runs_against.append(runs_1)
        SRH_balls_against.append(balls_1)


    if 'CSK' in m_df_2.bowling_team.unique():
        CSK_runs_against.append(runs_2)
        CSK_balls_against.append(balls_2)
    if 'DC' in m_df_2.bowling_team.unique():
        DC_runs_against.append(runs_2)
        DC_balls_against.append(balls_2)
    if 'GT' in m_df_2.bowling_team.unique():
        GT_runs_against.append(runs_2)
        GT_balls_against.append(balls_2)
    if 'KKR' in m_df_2.bowling_team.unique():
        KKR_runs_against.append(runs_2)
        KKR_balls_against.append(balls_2)
    if 'LSG' in m_df_2.bowling_team.unique():
        LSG_runs_against.append(runs_2)
        LSG_balls_against.append(balls_2)
    if 'MI' in m_df_2.bowling_team.unique():
        MI_runs_against.append(runs_2)
        MI_balls_against.append(balls_2)
    if 'PBKS' in m_df_2.bowling_team.unique():
        PBKS_runs_against.append(runs_2)
        PBKS_balls_against.append(balls_2)
    if 'RCB' in m_df_2.bowling_team.unique():
        RCB_runs_against.append(runs_2)
        RCB_balls_against.append(balls_2)
    if 'RR' in m_df_2.bowling_team.unique():
        RR_runs_against.append(runs_2)
        RR_balls_against.append(balls_2)
    if 'SRH' in m_df_2.bowling_team.unique():
        SRH_runs_against.append(runs_2)
        SRH_balls_against.append(balls_2)

CSK_runs_for = sum(CSK_runs_for)
CSK_balls_for = sum(CSK_balls_for)
CSK_runs_against = sum(CSK_runs_against)
CSK_balls_against = sum(CSK_balls_against)
DC_runs_for = sum(DC_runs_for)
DC_balls_for = sum(DC_balls_for)
DC_runs_against = sum(DC_runs_against)
DC_balls_against = sum(DC_balls_against)
GT_runs_for = sum(GT_runs_for)
GT_balls_for = sum(GT_balls_for)
GT_runs_against = sum(GT_runs_against)
GT_balls_against = sum(GT_balls_against)
KKR_runs_for = sum(KKR_runs_for)
KKR_balls_for = sum(KKR_balls_for)
KKR_runs_against = sum(KKR_runs_against)
KKR_balls_against = sum(KKR_balls_against)
LSG_runs_for = sum(LSG_runs_for)
LSG_balls_for = sum(LSG_balls_for)
LSG_runs_against = sum(LSG_runs_against)
LSG_balls_against = sum(LSG_balls_against)
MI_runs_for = sum(MI_runs_for)
MI_balls_for = sum(MI_balls_for)
MI_runs_against = sum(MI_runs_against)
MI_balls_against = sum(MI_balls_against)
PBKS_runs_for = sum(PBKS_runs_for)
PBKS_balls_for = sum(PBKS_balls_for)
PBKS_runs_against = sum(PBKS_runs_against)
PBKS_balls_against = sum(PBKS_balls_against)
RCB_runs_for = sum(RCB_runs_for)
RCB_balls_for = sum(RCB_balls_for)
RCB_runs_against = sum(RCB_runs_against)
RCB_balls_against = sum(RCB_balls_against)
RR_runs_for = sum(RR_runs_for)
RR_balls_for = sum(RR_balls_for)
RR_runs_against = sum(RR_runs_against)
RR_balls_against = sum(RR_balls_against)
SRH_runs_for = sum(SRH_runs_for)
SRH_balls_for = sum(SRH_balls_for)
SRH_runs_against = sum(SRH_runs_against)
SRH_balls_against = sum(SRH_balls_against)


teams = ["CSK", "DC", "GT","KKR","LSG","MI","PBKS","RCB","RR","SRH"]
pts_table_p2 = pd.DataFrame()
pts_table_p2['team'] = np.array(teams)
pts_table_p2['matches_played'] = 0

for index, row in pts_table_p2.iterrows():
    team = row['team']
    matches_played = df_all[df_all.batting_team==team].match_id.nunique()
    
    pts_table_p2.at[index, 'matches_played'] = matches_played

runs_for = pd.Series([CSK_runs_for, DC_runs_for, GT_runs_for, KKR_runs_for, LSG_runs_for,
           MI_runs_for, PBKS_runs_for, RCB_runs_for, RR_runs_for, SRH_runs_for])

balls_for = pd.Series([CSK_balls_for, DC_balls_for, GT_balls_for, KKR_balls_for, LSG_balls_for,
           MI_balls_for, PBKS_balls_for, RCB_balls_for, RR_balls_for, SRH_balls_for])

runs_against = pd.Series([CSK_runs_against, DC_runs_against, GT_runs_against, KKR_runs_against, LSG_runs_against,
           MI_runs_against, PBKS_runs_against, RCB_runs_against, RR_runs_against, SRH_runs_against])

balls_against = pd.Series([CSK_balls_against, DC_balls_against, GT_balls_against, KKR_balls_against, LSG_balls_against,
           MI_balls_against, PBKS_balls_against, RCB_balls_against, RR_balls_against, SRH_balls_against])

pts_table_p2['runs_for'] = runs_for
pts_table_p2['balls_for'] = balls_for

pts_table_p2['runs_against'] = runs_against
pts_table_p2['balls_against'] = balls_against

pts_table_p2['NRR'] = (6*pts_table_p2['runs_for']/pts_table_p2['balls_for']) - \
                                6*pts_table_p2['runs_against']/pts_table_p2['balls_against']


pts_table = pd.merge(left=pts_table, right=pts_table_p2, on='team', how='outer')
pts_table.fillna(0, inplace=True)
pts_table['points'] = pts_table['points'].astype(int)

pts_table = pts_table.sort_values(by=['points','NRR'], ascending=False).reset_index(drop=True)

# In[ ]:




