from hardcoded import *
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import datetime, timedelta
from dateutil.parser import parse
from os import path
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType
from bs4 import BeautifulSoup
import requests
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def attach_b2b_indicators():
	# Rewrites all_games.csv by attaching indicators for whether a played game is the second game of a 
	# back-to-back double header or not by retrieving schedule data from basketball_reference_web_scraper.
	
	all_games = pd.read_csv("./OutputCSVs/all_games.csv").reset_index()
	schedule = pd.DataFrame(client.season_schedule(season_end_year = 2017)) 
	for year in range(2018,2022):
		schedule = schedule.append(pd.DataFrame(client.season_schedule(season_end_year = year)))
	schedule["start_time"] = schedule["start_time"] + timedelta(hours = -7)
	schedule["Date"] = schedule["start_time"].apply(lambda x: x.strftime("%Y-%m-%d"))
	schedule["away_team"] = schedule["away_team"].apply(lambda x: x.value)
	schedule["home_team"] = schedule["home_team"].apply(lambda x: x.value)
	schedule["b2b_away"], schedule["b2b_home"] = [False] * len(schedule), [False] * len(schedule)
	all_games["away_team"], all_games["home_team"] = [""] * len(all_games), [""] * len(all_games)

	for team in all_abbrv:
		team_schedule = schedule.loc[(schedule["away_team"] == team) | (schedule["home_team"] == team)].reset_index()
		schedule.loc[team_schedule.loc[0, "index"],"b2b_home"] = False
		schedule.loc[team_schedule.loc[0, "index"],"b2b_away"] = False
		for i in range(len(team_schedule)-1):
			time = team_schedule.loc[i, "start_time"] + timedelta(hours = 24)
			if time.strftime("%Y-%m-%d") == team_schedule.loc[i+1, "Date"]:
				if team_schedule.loc[i+1, "home_team"] == team:
					schedule.loc[team_schedule.loc[i+1, "index"], "b2b_home"] = True
				else: 
					schedule.loc[team_schedule.loc[i+1, "index"], "b2b_away"] = True

	for i, row in all_games.iterrows():
		if row["location"] == "HOME": 
			all_games.at[i,"home_team"], all_games.at[i,"away_team"] = row["team"], row["opponent"]
		else: 
			all_games.at[i,"home_team"], all_games.at[i,"away_team"] = row["opponent"], row["team"]

	merged_df = all_games.merge(schedule, how="inner", on=["away_team","home_team","Date"])
	merged_df["b2b_indicator"] = [False] * len(merged_df)
	for i, row in merged_df.iterrows():
		if row["location"] == "HOME": 
			merged_df.at[i,"b2b_indicator"] = row["b2b_home"]
		else: 
			merged_df.at[i,"b2b_indicator"] = row["b2b_away"]
		
	merged_df.drop(merged_df.columns[range(24,31)], axis=1, inplace=True)
	merged_df = merged_df.sort_values("index").reset_index().iloc[:,3:]
	merged_df.to_csv("./OutputCSVs/all_games.csv")

def b2b_test():
	all_games = pd.read_csv("./OutputCSVs/all_games.csv").reset_index()
	schedule = pd.DataFrame(client.season_schedule(season_end_year = 2017)) 
	for year in range(2018,2022):
		schedule = schedule.append(pd.DataFrame(client.season_schedule(season_end_year = year)))
	schedule["start_time"] = schedule["start_time"] + timedelta(hours = -7)
	schedule["Date"] = schedule["start_time"].apply(lambda x: x.strftime("%Y-%m-%d"))
	schedule["away_team"] = schedule["away_team"].apply(lambda x: x.value)
	schedule["home_team"] = schedule["home_team"].apply(lambda x: x.value)
	schedule["b2b_away"], schedule["b2b_home"] = [False] * len(schedule), [False] * len(schedule)
	all_games["away_team"], all_games["home_team"] = [""] * len(all_games), [""] * len(all_games)

	for team in all_abbrv:
		team_schedule = schedule.loc[(df["away_team"] == team) | (schedule["home_team"] == team)]
		schedule.loc[team_schedule.loc[0, "index"],"b2b_home"] = False
		schedule.loc[team_schedule.loc[0, "index"],"b2b_away"] = False
		for i in range(len(team_schedule)-1):
			time = team_schedule.loc[i, "start_time"] + timedelta(hours = 24)
			if time.strftime("%Y-%m-%d") == team_schedule.loc[i+1, "Date"]:
				if team_schedule.loc[i+1, "home_team"] == team:
					schedule.loc[team_schedule.loc[i+1, "index"], "b2b_home"] = True
				else: 
					schedule.loc[team_schedule.loc[i+1, "index"], "b2b_away"] = True



def attach_team_stats():
	# This line gets the advanced stats from updated_team_stats.csv. This is something we scraped from 
	# NBA Advanced Stats and periodically updated. If we could streamline this process and replace it
	# with a function call that would be a better process.

	df = pd.read_csv("./OutputCSVs/updated_team_stats.csv")


	df["team"] = df["team"].str.upper()

	# This is the file we just wrote with all the scraped games.

	all_games = pd.read_csv("./OutputCSVs/all_games.csv")

	team_def = []
	team_pace = []
	team_tov = []
	opp_def = []
	opp_pace = []
	opp_tov = []

	# Here we attach team stats and opponent stats to each row of all_games.

	all_games_teams = all_games[["team", "opponent"]]

	for i in range(len(all_games_teams)):
		game = all_games_teams.loc[i]
		team = game["team"]
		opponent = game["opponent"]
		team_def.append(df[df["team"] == team]["drtg"].iloc[0])
		team_pace.append(df[df["team"] == team]["pace"].iloc[0])
		team_tov.append(df[df["team"] == team]["tov%"].iloc[0])
		opp_def.append(df[df["team"] == opponent]["drtg"].iloc[0])
		opp_pace.append(df[df["team"] == opponent]["pace"].iloc[0])
		opp_tov.append(df[df["team"] == opponent]["tov%"].iloc[0])
	
	all_games["Team Defensive Rating"] = team_def
	all_games["Team Pace"] = team_pace
	all_games["Team Turnover %"] = team_tov
	all_games["Opponent Defensive Rating"] = opp_def
	all_games["Opponent Pace"] = opp_pace
	all_games["Opponent Turnover %"] = opp_tov

	# I don't really know why we write all_games to file and then immediately
	# read the file into df. This could be an artifact of putting this together over
	# a couple months.

	df = all_games

	# Here we do a little processing because the basketball-reference data doesn't have certain stats.
	# If we could get NBA Advanced Stats game logs this is probably an avoidable step. Several points in
	# this process involve working around limitations in the bbref data that it would be nice to be
	# able to streamline away, especially if the backend is going to be public.

	attempted_2s = df["attempted_field_goals"] - df["attempted_three_point_field_goals"]
	made_2s = df["made_field_goals"] - df["made_three_point_field_goals"]
	rebounds = df["offensive_rebounds"] + df["defensive_rebounds"]
	at_home = df["location"] == "HOME"
	df["attempted_two_point_field_goals"] = attempted_2s
	df["made_two_point_field_goals"] = made_2s
	df["total_rebounds"] = rebounds
	df["at_home"] = at_home

	# And now we write this to file.

	df.to_csv("./OutputCSVs/all_games_updated.csv")

def get_stats(player, date, number_rows, start_date = False):
	"""Gets the last number_rows statlines from all_games_updated.csv for a player up to the given date. This
	gets called in generate_input_vector when we want to get a number of rows for this player to generate the
	weighted average statlines. Returns a pandas DataFrame of the desired statlines.
	
	Params:
	player: A string corresponding to entries in the 'name' column in all_games_updated.csv. Ex: 'LeBron James'
	date: A string YYYY-MM-DD. We parse this into a datetime object to compare it to the datetimes in
		all_games_updated.csv.
	number_rows: An integer number of rows to return.
	start_date: A string YYYY-MM-DD to start collecting data after.
	"""
	all_games_actual = pd.read_csv("./OutputCSVs/all_games_updated.csv")
	#all_games_actual = all_games_actual.iloc[:, 3:].reset_index()
	converted_datetime = datetime.strptime(date, '%Y-%m-%d')
	player_rows = all_games_actual.loc[all_games_actual['name'] == player]
	selected_rows = []
	if start_date:
		for i in range(len(player_rows)):
			this_date = datetime.strptime(player_rows.iloc[i]['Date'], '%Y-%m-%d')
			if this_date <= converted_datetime and this_date >= datetime.strptime(start_date, '%Y-%m-%d'):
				selected_rows.append(player_rows.iloc[i])
		return pd.DataFrame(selected_rows).sort_values(by=['Date'], ascending = False)
	if (len(player_rows)) < number_rows:
		for i in range(len(player_rows)):
			selected_rows.append(player_rows.iloc[i])
		return pd.DataFrame(selected_rows)
	index = 0
	for i in range(len(player_rows)):
		curr_date = player_rows.iloc[i]['Date']
		if datetime.strptime(curr_date, '%Y-%m-%d') >= converted_datetime:
			index = i
			break
	if index != 0:
		if index + 1 - number_rows < 0:
			for i in range(index + 1):
				selected_rows.append(player_rows.iloc[i])
		else:
			for i in range(index + 1 - number_rows, index + 1):
				selected_rows.append(player_rows.iloc[i])
	else:
		for i in range(len(player_rows) - number_rows, len(player_rows)):
			selected_rows.append(player_rows.iloc[len(player_rows) - i - 1])
	return pd.DataFrame(selected_rows).sort_values(by=['Date'], ascending = False)

def time_weighted_average(rows, statistic, weight):
	"""Takes in a set of rows, a given statistic to make the weighted average for, and a weight to build this average with.
	See documentation for weight_function above.

	Params:
	rows: The rows we are given to make a weighted average for.
	statistic: The statistic to create the weighted average for.
	weight: The weight we will use to make this average.
	"""
	def weight_function(statistic, weight):
		"""Takes in an array of a statistic and produces a weighted sum according to the weight.
		Honestly I am not very sure how this function works. I wrote it up at the same time as time_weighted_average
		and remember it being a fairly elegant solution to the problem of producing a weighted average with a bias towards
		recent games but it is not really obvious to me why this works.
		"""
		s = 0
		if type(statistic) == np.ndarray:
			for i in range(len(statistic)):
				s += statistic[len(statistic) - i - 1] * (weight ** i)
		else:
			for i in range(len(statistic)):
				s += statistic.iloc[len(statistic) - i - 1,] * (weight ** i)
		return s
	if rows.empty:
		return 0
	stat = rows[statistic]
	this_num = 1 / weight_function(np.ones(len(stat)), weight)
	return this_num * weight_function(stat, weight)

def add_team_defense(main_df):
	"""Takes in a DataFrame of statlines and tacks on the opponent defense against the position
	for each player, returning an augmented DataFrame. Searches the position for each player and finds the opponent's 
	rank against the position. This function as we applied it only used team defense against position as a 
	descriptive indicator in the frontend display - it didn't actually use the defense against position as a
	predictive input. It is probably adaptable to attach defense vs position to a set of inputs earlier in
	the pipeline so as to have predictive value in training the models.
	
	Params:
	main_df: A pandas DataFrame of players with positions, teams and opponents. As currently applied, this
		is a DataFrame of created predictions, the hope is that this works with a DataFrame of model inputs so
		that the defense vs position can be an actual model input.
	"""

	# This is a csv of player positions. This needs to be updated every so often but
	# probably does not have to be automated.

	positions = pd.read_csv('./OutputCSVs/all_player_positions.csv')
	team_def_vs_pos = []
	opp_def_vs_pos = []

	dbp_2017 = pd.read_csv('./OutputCSVs/team_def_vs_position_2017.csv')
	dbp_2018 = pd.read_csv('./OutputCSVs/team_def_vs_position_2018.csv')
	dbp_2019 = pd.read_csv('./OutputCSVs/team_def_vs_position_2019.csv')
	dbp_2020 = pd.read_csv('./OutputCSVs/team_def_vs_position_2020.csv')
	dbp_2021 = pd.read_csv('./OutputCSVs/team_def_vs_position_2021.csv')

	player_positions = pd.Series(positions['position'].values,index=positions['player name']).to_dict()
	
	def process_dual_positions(position1, position2, team, opponent, dbp):
		"""Given a position for a player on a team and a position for a player on the opposing team,
		this function searches the defense vs position csv for each team's rank against the other
		position. Samay wrote this one and I don't really know all the details about how it works.
	
		Params:
		position1: A string (?) representing the position for the player on 'team'.
		position2: A string (?) representing the position for the player on 'opponent'.
		team: The team that the player in position1 plays for, corresponding to the 'Team' column in the
			defense by position csv.
		opponent: The team that the player in position2 plays for, corresponding to the 'Team' column in the
			defense by position csv.
		dbp: CSV of team defense by position.

		"""
		first_team_subrank =  dbp.loc[dbp['Team']==team, 'vs {0}'.format(position1)].iloc[0]
		second_team_subrank =  dbp.loc[dbp['Team']==team, 'vs {0}'.format(position2)].iloc[0]
		first_opp_subrank =  dbp.loc[dbp['Team']==opponent, 'vs {0}'.format(position1)].iloc[0]
		second_opp_subrank =  dbp.loc[dbp['Team']==opponent, 'vs {0}'.format(position2)].iloc[0]
		return first_team_subrank, second_team_subrank, first_opp_subrank, second_opp_subrank

	for i in range(len(main_df)):
		name = main_df['name'].iloc[i]
		date = main_df['date'].iloc[i]
		if int(date[:4]) == 2021:
			dbp = dbp_2021
		elif int(date[:4]) == 2020:
			if int(date[5:7]) > 10:
				dbp = dbp_2021
			else:
				dbp = dbp_2020
		elif int(date[:4]) == 2019:
			if int(date[5:7]) > 9:
				dbp = dbp_2020
			else:
				dbp = dbp_2019
		elif int(date[:4]) == 2018:
			if int(date[5:7]) > 9:
				dbp = dbp_2019
			else:
				dbp = dbp_2018
		elif int(date[:4]) == 2017:
			if int(date[5:7]) > 9:
				dbp = dbp_2018
			else:
				dbp = dbp_2017
		else:
			dbp = dbp_2017
		position = player_positions.get(name)
		team = all_abbrv.get(main_df['team'].iloc[i])
		opponent = all_abbrv.get(main_df['opponent'].iloc[i])
		if position is None:
			team_def_vs_pos.append(15.5)
			opp_def_vs_pos.append(15.5)
		else:
			if position in ['PG','SG','SF','PF','C']:
				team_def_vs_pos.append(dbp.loc[dbp['Team']==team, 'vs {0}'.format(position)].iloc[0])
				opp_def_vs_pos.append(dbp.loc[dbp['Team']==opponent, 'vs {0}'.format(position)].iloc[0])
			elif position == 'G':
				pdp = process_dual_positions('PG', 'SG', team, opponent, dbp)
				team_def_vs_pos.append((pdp[0] + pdp[1])/2)
				opp_def_vs_pos.append((pdp[2] + pdp[3])/2)
			elif position == 'F':
				pdp = process_dual_positions('SF', 'PF', team, opponent, dbp)
				team_def_vs_pos.append((pdp[0] + pdp[1])/2)
				opp_def_vs_pos.append((pdp[2] + pdp[3])/2)
			elif position in ['G-F','F-G']:
				pdp = process_dual_positions('SG', 'SF', team, opponent, dbp)
				team_def_vs_pos.append((pdp[0] + pdp[1])/2)
				opp_def_vs_pos.append((pdp[2] + pdp[3])/2)
			elif position in ['F-C']:
				pdp = process_dual_positions('PF', 'C', team, opponent, dbp)
				team_def_vs_pos.append((pdp[0] + pdp[1])/2)
				opp_def_vs_pos.append((pdp[2] + pdp[3])/2)
	#main_df['team def vs pos'] = team_def_vs_pos
	main_df['Opponent Defensive Rank vs Position'] = opp_def_vs_pos
	return main_df

def add_over_under(main_df):
	ou_2017 = pd.read_csv("./OutputCSVs/2016-17_OU.csv")
	ou_2018 = pd.read_csv("./OutputCSVs/2017-18_OU.csv")
	ou_2019 = pd.read_csv("./OutputCSVs/2018-19_OU.csv")
	ou_2020 = pd.read_csv("./OutputCSVs/2019-20_OU.csv")
	ou_2021 = pd.read_csv("./OutputCSVs/2020-21_OU.csv")
	ou = ou_2017.append(ou_2018).append(ou_2019).append(ou_2020).append(ou_2021)
	ou.reset_index()
	ou["year"] = ou["date"]//10000
	ou["strdate"] = ou["date"].astype(str)
	ou = ou[~ou.strdate.str.contains("202010")]
	main_df["newdate"] = main_df["date"].str.replace("-", "")
	over_under = []
	for i in main_df.index:
		newdate = int(main_df.loc[i, "newdate"])
		newyear = newdate // 10000
		location = main_df.loc[i, "location"]
		team = betting_dictionary.get(main_df['team'][i])
		opponent = betting_dictionary.get(main_df['opponent'][i])
		dates = ou["strdate"].unique()
		if newdate in dates:
			if i % 500 == 0:
				print(newdate)
			if location == "AWAY":
				over_under.append(ou.loc[(ou["date"] == newdate) & (ou["o:team"] == opponent), "total"].values[0])
			else:
				over_under.append(ou.loc[(ou["date"] == newdate) & (ou["team"] == team), "total"].values[0])
		else:
			over_under.append(np.mean(ou.loc[(ou["team"] == team) & (ou["year"] == newyear), "total"]))
	main_df = main_df.drop(["newdate"], axis = 1)
	main_df["total"] = over_under
	return main_df

def add_isolation_offense(main_df):
	playeriso_2017 = pd.read_csv("./IsolationStats/PlayerIsolationOffense/CSVs/2016-17 Player Isolation Offense.csv")
	playeriso_2017["Year"] = [2017 for _ in range(playeriso_2017.shape[0])]
	playeriso_2018 = pd.read_csv("./IsolationStats/PlayerIsolationOffense/CSVs/2017-18 Player Isolation Offense.csv")
	playeriso_2018["Year"] = [2018 for _ in range(playeriso_2018.shape[0])]
	playeriso_2019 = pd.read_csv("./IsolationStats/PlayerIsolationOffense/CSVs/2018-19 Player Isolation Offense.csv")
	playeriso_2019["Year"] = [2019 for _ in range(playeriso_2019.shape[0])]
	playeriso_2020 = pd.read_csv("./IsolationStats/PlayerIsolationOffense/CSVs/2019-20 Player Isolation Offense.csv")
	playeriso_2020["Year"] = [2020 for _ in range(playeriso_2020.shape[0])]
	playeriso_2021 = pd.read_csv("./IsolationStats/PlayerIsolationOffense/CSVs/2020-21 Player Isolation Offense.csv")
	playeriso_2021["Year"] = [2021 for _ in range(playeriso_2021.shape[0])]

	playeriso = playeriso_2017.append(playeriso_2018).append(playeriso_2019).append(playeriso_2020).append(playeriso_2021)
	playeriso.reset_index()
	main_df["newdate"] = main_df["date"].str.replace("-", "")
	poss = []
	ppp = []
	fga = []
	for i in main_df.index:
		newdate = int(main_df.loc[i, "newdate"])
		playername = main_df.loc[i, "name"]
		newyear = newdate // 10000
		if (newdate // 100) - (100 * newyear) > 10:
			newyear = newyear + 1
		iso_row = playeriso.loc[(playeriso["Year"] == newyear) & (playeriso["PLAYER"] == playername), ["POSS", "PPP", "FGA"]]
		if iso_row.shape[0] > 0:
			poss.append(iso_row["POSS"].values[0])
			ppp.append(iso_row["PPP"].values[0])
			fga.append(iso_row["FGA"].values[0])
		else:
			poss.append(0)
			ppp.append(0)
			fga.append(0)
	main_df = main_df.drop(["newdate"], axis = 1)
	main_df["Iso POSS"] = poss
	main_df["Iso PPP"] = ppp
	main_df["Iso FGA"] = fga
	return main_df

def add_isolation_defense(main_df):
	teamdiso_2017 = pd.read_csv("./IsolationStats/TeamIsolationDefense/Defense/2016-17 Team Isolation Defense.csv")
	teamdiso_2017["Year"] = [2017 for _ in range(teamdiso_2017.shape[0])]
	teamdiso_2018 = pd.read_csv("./IsolationStats/TeamIsolationDefense/Defense/2017-18 Team Isolation Defense.csv")
	teamdiso_2018["Year"] = [2018 for _ in range(teamdiso_2018.shape[0])]
	teamdiso_2019 = pd.read_csv("./IsolationStats/TeamIsolationDefense/Defense/2018-19 Team Isolation Defense.csv")
	teamdiso_2019["Year"] = [2019 for _ in range(teamdiso_2019.shape[0])]
	teamdiso_2020 = pd.read_csv("./IsolationStats/TeamIsolationDefense/Defense/2019-20 Team Isolation Defense.csv")
	teamdiso_2020["Year"] = [2020 for _ in range(teamdiso_2020.shape[0])]
	teamdiso_2021 = pd.read_csv("./IsolationStats/TeamIsolationDefense/Defense/2020-21 Team Isolation Defense.csv")
	teamdiso_2021["Year"] = [2021 for _ in range(teamdiso_2021.shape[0])]

	teamdiso = teamdiso_2017.append(teamdiso_2018).append(teamdiso_2019).append(teamdiso_2020).append(teamdiso_2021)
	teamdiso.reset_index()

	main_df["newdate"] = main_df["date"].str.replace("-", "")
	main_df["opponent"] = main_df["opponent"].str.lower()
	teamdiso["TEAM"] = teamdiso["TEAM"].str.lower()
	teamdiso["SCORE FREQ"] = teamdiso["SCORE FREQ"].astype(str)
	teamdiso["SCORE FREQ"] = teamdiso["SCORE FREQ"].str.rstrip('%').astype('float')
	teamdiso["SCORE FREQ"] = teamdiso["SCORE FREQ"] / 100
	poss = []
	ppp = []
	fga = []
	freq = []
	for i in main_df.index:
		newdate = int(main_df.loc[i, "newdate"])
		opponent = main_df.loc[i, "opponent"]
		newyear = newdate // 10000
		if (newdate // 100) - (100 * newyear) > 10:
			newyear = newyear + 1
		iso_row = teamdiso.loc[(teamdiso["Year"] == newyear) & (teamdiso["TEAM"] == opponent), ["POSS", "PPP", "FGA", "SCORE FREQ"]]
		if iso_row.shape[0] > 0:
			poss.append(iso_row["POSS"].values[0])
			ppp.append(iso_row["PPP"].values[0])
			fga.append(iso_row["FGA"].values[0])
			freq.append(iso_row["SCORE FREQ"].values[0])
		else:
			poss.append(0)
			ppp.append(0)
			fga.append(0)
			freq.append(np.mean(freq))
	main_df = main_df.drop(["newdate"], axis = 1)
	main_df["Opp D Iso POSS"] = poss
	main_df["Opp D Iso PPP"] = ppp
	main_df["Opp D Iso FGA"] = fga
	main_df["Opp D Iso Score %"] = freq
	main_df = main_df.round(2)
	main_df["opponent"] = main_df["opponent"].str.upper()
	return main_df

def add_team_isolation_offense(main_df):
	teamoiso_2017 = pd.read_csv("./IsolationStats/TeamIsolationOffense/Offense/2016-17 Team Isolation Offense.csv")
	teamoiso_2017["Year"] = [2017 for _ in range(teamoiso_2017.shape[0])]
	teamoiso_2018 = pd.read_csv("./IsolationStats/TeamIsolationOffense/Offense/2017-18 Team Isolation Offense.csv")
	teamoiso_2018["Year"] = [2018 for _ in range(teamoiso_2018.shape[0])]
	teamoiso_2019 = pd.read_csv("./IsolationStats/TeamIsolationOffense/Offense/2018-19 Team Isolation Offense.csv")
	teamoiso_2019["Year"] = [2019 for _ in range(teamoiso_2019.shape[0])]
	teamoiso_2020 = pd.read_csv("./IsolationStats/TeamIsolationOffense/Offense/2019-20 Team Isolation Offense.csv")
	teamoiso_2020["Year"] = [2020 for _ in range(teamoiso_2020.shape[0])]
	teamoiso_2021 = pd.read_csv("./IsolationStats/TeamIsolationOffense/Offense/2020-21 Team Isolation Offense.csv")
	teamoiso_2021["Year"] = [2021 for _ in range(teamoiso_2021.shape[0])]

	teamoiso = teamoiso_2017.append(teamoiso_2018).append(teamoiso_2019).append(teamoiso_2020).append(teamoiso_2021)
	teamoiso.reset_index()
	main_df["newdate"] = main_df["date"].str.replace("-", "")
	main_df["team"] = main_df["team"].str.lower()
	teamoiso["TEAM"] = teamoiso["TEAM"].str.lower()
	teamoiso["SCORE FREQ"] = teamoiso["SCORE FREQ"].astype(str)
	teamoiso["SCORE FREQ"] = teamoiso["SCORE FREQ"].str.rstrip('%').astype('float')
	teamoiso["SCORE FREQ"] = teamoiso["SCORE FREQ"] / 100
	poss = []
	ppp = []
	fga = []
	freq = []
	for i in main_df.index:
		newdate = int(main_df.loc[i, "newdate"])
		team = main_df.loc[i, "team"]
		newyear = newdate // 10000
		if (newdate // 100) - (100 * newyear) > 10:
			newyear = newyear + 1
		iso_row = teamoiso.loc[(teamoiso["Year"] == newyear) & (teamoiso["TEAM"] == team), ["POSS", "PPP", "FGA", "SCORE FREQ"]]
		if iso_row.shape[0] > 0:
			poss.append(iso_row["POSS"].values[0])
			ppp.append(iso_row["PPP"].values[0])
			fga.append(iso_row["FGA"].values[0])
			freq.append(iso_row["SCORE FREQ"].values[0])
		else:
			poss.append(0)
			ppp.append(0)
			fga.append(0)
			freq.append(np.mean(freq))
	main_df = main_df.drop(["newdate"], axis = 1)
	main_df["Team Iso POSS"] = poss
	main_df["Team Iso PPP"] = ppp
	main_df["Team Iso FGA"] = fga
	main_df["Team Iso Score %"] = freq
	main_df = main_df.round(2)
	main_df["team"] = main_df["team"].str.upper()
	return main_df

def add_rate_statistics(main_df):
	team_box_scores = pd.read_csv("./TeamBoxScores/all_box_scores.csv")
	main_df["newdate"] = main_df["date"].str.replace("-", "").astype(int)
	main_df["year"] = main_df["newdate"] // 10000
	main_df["month"] = (main_df["newdate"] - 10000 * main_df["year"]) // 100
	main_df["day"] = main_df["newdate"] - 10000 * main_df["year"] - 100 * main_df["month"]
	usage = []
	orebrate = []
	drebrate = []
	rebrate = []
	pie = []
	for i in main_df.index:
		t = main_df.loc[i, "team"]
		o = main_df.loc[i, "opponent"]
		y = main_df.loc[i, "year"]
		m = main_df.loc[i, "month"]
		d = main_df.loc[i, "day"]
		game_row = team_box_scores.loc[(team_box_scores["Year"] == y) & 
									   (team_box_scores["Month"] == m) & 
									   (team_box_scores["Day"] == d) & 
									   (team_box_scores["team"] == t), 
									   ["minutes_played", "made_field_goals", "attempted_field_goals", "made_three_point_field_goals",
									   "attempted_three_point_field_goals", "made_free_throws", "attempted_free_throws",
									   "offensive_rebounds", "defensive_rebounds", "assists", "steals", "blocks",
									   "turnovers", "personal_fouls"]]
		opponent_row = team_box_scores.loc[(team_box_scores["Year"] == y) & 
									   (team_box_scores["Month"] == m) & 
									   (team_box_scores["Day"] == d) & 
									   (team_box_scores["team"] == o), 
									   ["offensive_rebounds", "defensive_rebounds"]]
		
		if len(game_row["minutes_played"].values) > 0:
			usage.append((100 * ((main_df.loc[i, "attempted_two_point_field_goals"] + main_df.loc[i, "attempted_three_point_field_goals"]) +
						.44 * main_df.loc[i, "attempted_free_throws"] + 
							 main_df.loc[i, "turnovers"]) * 
					  game_row["minutes_played"].values[0]) /
					((game_row["attempted_field_goals"].values[0] + .44 * game_row["attempted_free_throws"].values[0] + game_row["turnovers"].values[0]) *
					 (main_df.loc[i, "seconds_played"] / 12)))
			orebrate.append((20 * (main_df.loc[i, "offensive_rebounds"]) * game_row["minutes_played"].values[0]) /
					   ((main_df.loc[i, "seconds_played"] / 60) * (game_row["offensive_rebounds"].values[0] + opponent_row["defensive_rebounds"].values[0])))
			drebrate.append((20 * (main_df.loc[i, "defensive_rebounds"]) * game_row["minutes_played"].values[0]) /
					   ((main_df.loc[i, "seconds_played"] / 60) * (opponent_row["offensive_rebounds"].values[0] + game_row["defensive_rebounds"].values[0])))
			rebrate.append((20 * (main_df.loc[i, "offensive_rebounds"] + main_df.loc[i, "defensive_rebounds"]) * game_row["minutes_played"].values[0]) /
					   ((main_df.loc[i, "seconds_played"] / 60) * (opponent_row["offensive_rebounds"].values[0] + opponent_row["defensive_rebounds"].values[0] + game_row["offensive_rebounds"].values[0] + game_row["defensive_rebounds"].values[0])))
			pie.append((3 * main_df.loc[i, "made_two_point_field_goals"] + 4 * main_df.loc[i, "made_three_point_field_goals"] + 2 * main_df.loc[i, "made_free_throws"] - 
					main_df.loc[i, "attempted_two_point_field_goals"] - main_df.loc[i, "attempted_three_point_field_goals"] - main_df.loc[i, "attempted_free_throws"] +
					main_df.loc[i, "defensive_rebounds"] + .5 * main_df.loc[i, "offensive_rebounds"] + main_df.loc[i, "assists"] +
					main_df.loc[i, "steals"] + .5 * main_df.loc[i, "blocks"] - main_df.loc[i, "turnovers"]) /
				  (3 * (game_row["made_field_goals"].values[0] - game_row["made_three_point_field_goals"].values[0]) + 4 * game_row["made_three_point_field_goals"].values[0] + 2 * game_row["made_free_throws"].values[0] -
				  game_row["attempted_field_goals"].values[0] - game_row["attempted_free_throws"].values[0] + game_row["defensive_rebounds"].values[0] + .5 * game_row["offensive_rebounds"].values[0] +
				  game_row["assists"].values[0] + game_row["steals"].values[0] + .5 * game_row["blocks"].values[0] - game_row["turnovers"].values[0]))
		else:
			usage.append(0)
			orebrate.append(0)
			drebrate.append(0)
			rebrate.append(0)
			pie.append(0)
		
	main_df = main_df.drop(["newdate", "year", "month", "day"], axis = 1)
	main_df["Usage Rate"] = usage
	main_df["OReb %"] = orebrate
	main_df["DReb %"] = drebrate
	main_df["Reb %"] = rebrate
	main_df["PIE"] = pie
	main_df = main_df.round(2)
	return main_df


def generate_input_vector(player_box_scores, input_statistics, sample_size = 5, weight = .8, per_min = False):
	"""Takes in box scores, an array of input statistics, a sample size of games, and a weight.
	Produces, for each player in player_box_scores, a weighted average of each statistic in input_statistics
	over the preceding sample_size games for that player with the specified weight. So if sample_size is 5, 
	the first 5 statlines in generate_input_vector for each player will have incomplete versions of the weighted average.
	Because time_weighted_average adjusts for this there won't be any wacky numbers produced for those rows,
	but they will be very biased towards the most recent games in the sample as they aren't producing weighted averages
	over the full desired length of the sample window.
	
	This function generates the inputs for our models. Because of the above weirdness it is good to get very large sample
	sizes here so the first few rows don't end up biasing the models too much. I am pretty sure the object it returns is
	a pandas DataFrame.
	
	Params:
	player_box_scores: A pandas DataFrame of game statlines for each player. It has both the bbref box score stats
		and the NBA Advanced Team stats we helpfully tacked on earlier.
	input_statistics: An array of strings representing the columns of player_box_scores we want to create weighted
		averages of.
	sample_size: The number of rows we will use to create weighted averages. This is an input to time_weighted_average.
	weight: The weight we will use to create weighted averages. This is an input to time_weighted_average.
	"""
	print("its working I think")
	if per_min:
		return generate_input_vector_per_min(player_box_scores, input_statistics, sample_size, weight)
	player_box_scores = player_box_scores[~player_box_scores.index.duplicated()]
	player_box_scores.reindex(range(len(player_box_scores)), axis = "index")
	predicted_statlines = pd.DataFrame(index = player_box_scores.index, columns = input_statistics).fillna(0).T
	index_len = len(player_box_scores.index)
	for box_index in player_box_scores.index:
		box_score = player_box_scores.loc[box_index]
		player_name = box_score["name"]
		game_date = str(box_score["Date"])[:10]
		last_n_rows = get_stats(player_name, game_date, sample_size)
		weighted_stats = [player_name, box_score["team"], game_date, box_score["location"], box_score["opponent"]]
		for stat in input_statistics[5:]:
			weighted_stats.append(round(time_weighted_average(last_n_rows, stat, weight), 2))
		predicted_statlines[box_index] = weighted_stats
	inputs = predicted_statlines.T
	inputs = add_team_defense(inputs)
	inputs = add_over_under(inputs)
	inputs = add_isolation_offense(inputs)
	inputs = add_isolation_defense(inputs)
	inputs = add_team_isolation_offense(inputs)
	inputs = add_rate_statistics(inputs)
	return inputs

def generate_input_vector_per_min(player_box_scores, input_statistics, sample_size = 5, weight = .8):
	"""Takes in box scores, an array of input statistics, a sample size of games, and a weight.
	Produces, for each player in player_box_scores, a weighted average of each statistic in input_statistics
	over the preceding sample_size games for that player with the specified weight. So if sample_size is 5, 
	the first 5 statlines in generate_input_vector for each player will have incomplete versions of the weighted average.
	Because time_weighted_average adjusts for this there won't be any wacky numbers produced for those rows,
	but they will be very biased towards the most recent games in the sample as they aren't producing weighted averages
	over the full desired length of the sample window.
	
	This function generates the inputs for our models. Because of the above weirdness it is good to get very large sample
	sizes here so the first few rows don't end up biasing the models too much. I am pretty sure the object it returns is
	a pandas DataFrame.
	
	Params:
	player_box_scores: A pandas DataFrame of game statlines for each player. It has both the bbref box score stats
		and the NBA Advanced Team stats we helpfully tacked on earlier.
	input_statistics: An array of strings representing the columns of player_box_scores we want to create weighted
		averages of.
	sample_size: The number of rows we will use to create weighted averages. This is an input to time_weighted_average.
	weight: The weight we will use to create weighted averages. This is an input to time_weighted_average.
	"""
	player_box_scores = player_box_scores[~player_box_scores.index.duplicated()]
	player_box_scores.reindex(range(len(player_box_scores)), axis = "index")
	predicted_statlines = pd.DataFrame(index = player_box_scores.index, columns = input_statistics).fillna(0).T
	index_len = len(player_box_scores.index)
	for box_index in player_box_scores.index:
		box_score = player_box_scores.loc[box_index]
		player_name = box_score["name"]
		game_date = str(box_score["Date"])[:10]
		last_n_rows = get_stats(player_name, game_date, sample_size)
		weighted_stats = [player_name, box_score["team"], game_date, box_score["location"], box_score["opponent"]]
		recent_minutes = time_weighted_average(last_n_rows, "seconds_played", weight)/60
		mins = player_box_scores.loc[box_index, "seconds_played"]
		for i in range(5, len(input_statistics)):
			stat = input_statistics[i]
			if recent_minutes > 0 and i < (len(input_statistics) - 6):
				weighted_stats.append(round(time_weighted_average(last_n_rows, stat, weight)/recent_minutes, 2))
			else:
				weighted_stats.append(round(time_weighted_average(last_n_rows, stat, weight), 2))
		predicted_statlines[box_index] = weighted_stats
	inputs = predicted_statlines.T
	inputs = add_team_defense(inputs)
	inputs = add_over_under(inputs)
	inputs = add_isolation_offense(inputs)
	inputs = add_isolation_defense(inputs)
	inputs = add_team_isolation_offense(inputs)
	inputs = add_rate_statistics(inputs)

	#THIS ONE IS THE PER MINUTE ONE

	return inputs


def n_game_average(player, date, sample_size, start_date = False):
	"""Takes in a player, date, sample size of games, and optional date to start collecting from.
	Calls get_stats to return the player's statlines in this range and returns the average statline
	for the player in each category over the desired range of games.
	
	Params:
	player: A string corresponding to entries in the 'name' column in all_games_updated.csv. Ex: 'LeBron James'
	date: A string YYYY-MM-DD. We parse this into a datetime object to compare it to the datetimes in
		all_games_updated.csv.
	sample_size: An integer number of rows to return.
	start_date: A string YYYY-MM-DD to start collecting data after. 
	"""
	season_stats = get_stats(player, date, sample_size, start_date)
	season_stats["is_win"] = season_stats["outcome"] == "WIN"
	cols_to_note = list(season_stats.loc[season_stats.index[0], ["name", "team"]])
	averages = [np.mean(season_stats[col]) for col in cols_to_average]
	return cols_to_note + averages

def season_average(player, year):
	"""Takes in a player and year and returns the average statline for the player in each category
	for games in the season ending that year.
	
	Params:
	player: A string corresponding to entries in the 'name' column in all_games_updated.csv. Ex: 'LeBron James'
	year: A number YYYY.
	"""
	start_date = str(year - 1) + "-10-15"
	if year == 2020:
		end_date = str(year) + "-08-16"
	if year == 2021:
		end_date = str(year) + "-05-10"
	else:
		end_date = str(year) + "-04-17"
	return n_game_average(player, end_date, 82, start_date)

def generate_input_matrix(player, date, sample_size):
	"""Takes in a player, date and sample size of games. Returns a matrix to be used for model input,
	with rows for each of the following inputs:
		-The player's season average statline across each category.
		-The player's previous average statline across each category.
		-The player's 15-game average statline across each category.
		-The player's 10-game average statline across each category.
		-The player's 5-game average statline across each category.
		-The player's game log in each category in the last sample_size games.
	
	Params:
	player: A string corresponding to entries in the 'name' column in all_games_updated.csv. Ex: 'LeBron James'
	date: A string YYYY-MM-DD. We parse this into a datetime object to compare it to the datetimes in
		all_games_updated.csv.
	sample_size: An integer number of rows to return.
	"""
	date = str(date)[:10]
	last_n_rows = get_stats(player, date, sample_size)
	last_n_rows["is_win"] = last_n_rows["outcome"] == "WIN"
	last_n_rows["date"] = last_n_rows["Date"].astype(str)
	last_n_rows = add_team_defense(last_n_rows)
	last_n_rows = add_over_under(last_n_rows)
	last_n_rows = add_isolation_offense(last_n_rows)
	last_n_rows = add_isolation_defense(last_n_rows)
	last_n_rows = add_team_isolation_offense(last_n_rows)
	last_n_rows = add_rate_statistics(last_n_rows)
	this_season_average = season_average(player, int(date[:4]))
	last_season_average = season_average(player, int(date[:4]) - 1)
	l15_average = n_game_average(player, date, 15)
	l10_average = n_game_average(player, date, 10)
	l5_average = n_game_average(player, date, 5)
	for average in [this_season_average, last_season_average, l15_average, l10_average, l5_average]:
		count = 0
		array_to_append = []
		for c in last_n_rows.columns:
			if c in ["name", "team"] or c in cols_to_average:
				array_to_append.append(average[count])
				count = count + 1
			else:
				array_to_append.append(0)
		array_to_append = pd.Series(array_to_append, index = last_n_rows.columns)
		last_n_rows = last_n_rows.append(array_to_append, ignore_index = True)
	last_n_rows = last_n_rows.drop(columns = ["Unnamed: 0", "Unnamed: 0.1"])
	return last_n_rows

def input_matrix_to_tensor(input_matrix):
	input_columns_to_keep = ['wl',
	   'seconds_played', 'made_field_goals', 'attempted_field_goals',
	   'made_three_point_field_goals', 'attempted_three_point_field_goals',
	   'made_free_throws', 'attempted_free_throws', 'offensive_rebounds',
	   'defensive_rebounds', 'assists', 'steals', 'blocks', 'turnovers',
	   'personal_fouls', 'game_score', 'b2b_indicator',
	   'Team Defensive Rating', 'Team Pace', 'Team Turnover %',
	   'Opponent Defensive Rating', 'Opponent Pace', 'Opponent Turnover %',
	   'attempted_two_point_field_goals', 'made_two_point_field_goals',
	   'total_rebounds', 'at_home', 'is_win',
	   'Opponent Defensive Rank vs Position', 'total', 'Iso POSS', 'Iso PPP',
	   'Iso FGA', 'Opp D Iso POSS', 'Opp D Iso PPP', 'Opp D Iso FGA',
	   'Opp D Iso Score %', 'Team Iso POSS', 'Team Iso PPP', 'Team Iso FGA',
	   'Team Iso Score %', 'Usage Rate', 'OReb %', 'DReb %', 'Reb %', 'PIE']
	input_matrix["winner"] = input_matrix["outcome"] == "WIN"
	input_matrix["loser"] = input_matrix["outcome"] == "LOSS"
	input_matrix["wl"] = input_matrix["winner"].astype(int) - input_matrix["loser"].astype(int)
	input_matrix = input_matrix[input_columns_to_keep]
	print(input_matrix.shape)
	input_matrix = np.array(input_matrix)
	input_tensor = tf.convert_to_tensor(input_matrix, dtype=tf.float32)
	return input_tensor