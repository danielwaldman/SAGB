from hardcoded import *
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType
import numpy as np
import pandas as pd
from os import path
from datetime import datetime

def double_double(threes, twos, fts, rebounds, assists):
	"""Returns whether the input values for a given statline constitute a double_double.
	Inputs should already be floats but have caused trouble in unexpected spots.
	
	TODO:
		-Investigate whether the .tolist() calls are significantly wasting our time and if they
		can be removed. This function is only really called at the end of the pipeline but this could
		be a time waster."""
	points = float(threes) * 3 + float(twos) * 2 + float(fts)
	rebounds = rebounds.tolist()[0]
	assists = assists.tolist()[0]
	return (points >= 10 and rebounds >= 10) or (points >= 10 and assists >= 10) or (rebounds >= 10 and assists >= 10)

def triple_double(threes, twos, fts, rebounds, assists):
	"""See above documentation for double_double. Returns whether statline constitutes a triple_double.
	Similar concerns regarding input types."""
	points = float(threes) * 3 + float(twos) * 2 + float(fts)
	rebounds = rebounds.tolist()[0]
	assists = assists.tolist()[0]
	return points >= 10 and rebounds >= 10 and assists >= 10

def get_points(row_data):
	"""Returns a tuple of Fanduel points and equivalent Fanduel dollar value for a given statline.
	This gets called when we want to return a number of points for a statline.
	
	Params:
	row_data: One row of a pandas DataFrame. May or may not have the columns attribute, which we look for
		just in case.
	"""
	if 'made_three_point_field_goals_y' in row_data.columns:
		three_pt_fgs = row_data['made_three_point_field_goals_y']
		two_pt_fgs = row_data['made_two_point_field_goals_y']
		made_fts = row_data['made_free_throws_y']
		total_rebounds = row_data['rebounds_y']
		assists = row_data['assists_y']
		blocks = row_data['blocks_y']
		steals = row_data['steals_y']
		turnovers = row_data['turnovers_y']
	else:
		three_pt_fgs = row_data['made_three_point_field_goals']
		two_pt_fgs = row_data['made_two_point_field_goals']
		made_fts = row_data['made_free_throws']
		total_rebounds = row_data['rebounds']
		assists = row_data['assists']
		blocks = row_data['blocks']
		steals = row_data['steals']
		turnovers = row_data['turnovers']
	FD_points = three_pt_fgs * 3 + two_pt_fgs * 2 + made_fts + total_rebounds * 1.2 + assists * 1.5 + blocks * 3 + steals * 3 - turnovers
	FD_dollars = FD_points * 200
	return (FD_points, FD_dollars)

def get_draftkings_points(row_data):
	"""See above documentation for get_points. Returns a tuple of Draftkings points and dollar value based on DK
	scoring rules and relative values. Coerces some values to floats to be able to ensure double_double and triple_double
	functions work smoothly.
	
	Params:
	row_data: One row of a pandas DataFrame. May or may not have the columns attribute, which we look for
		just in case.
	"""
	if 'made_three_point_field_goals_y' in row_data.columns:
		three_pt_fgs = float(row_data['made_three_point_field_goals_y'])
		two_pt_fgs = float(row_data['made_two_point_field_goals_y'])
		made_fts = float(row_data['made_free_throws_y'])
		total_rebounds = float(row_data['rebounds_y'])
		assists = float(row_data['assists_y'])
		blocks = row_data['blocks_y']
		steals = row_data['steals_y']
		turnovers = row_data['turnovers_y']
	else:
		three_pt_fgs = float(row_data['made_three_point_field_goals'])
		two_pt_fgs = float(row_data['made_two_point_field_goals'])
		made_fts = float(row_data['made_free_throws'])
		total_rebounds = row_data['rebounds']
		assists = row_data['assists']
		blocks = row_data['blocks']
		steals = row_data['steals']
		turnovers = row_data['turnovers']
	DK_points = three_pt_fgs * 3.5 + two_pt_fgs * 2 + made_fts + total_rebounds * 1.25 + assists * 1.5 + blocks * 2 + steals * 2 - .5 * turnovers + 1.5 * double_double(three_pt_fgs, two_pt_fgs, made_fts, total_rebounds, assists) + 3 * triple_double(three_pt_fgs, two_pt_fgs, made_fts, total_rebounds, assists)
	DK_dollars = DK_points * 187.5
	return (DK_points, DK_dollars)


def minutes_predictor(weighted_stats):
	"""Produces the minutes projections for a set of weighted statlines. Just returns the mean weighted seconds
	in the given rows. Right now this function doesn't actually get called anywhere, but if we come up with a way to
	get better minutes projections out of a set of weighted stats we can reimplement it.
	
	Params:
	weighted_stats: A pandas DataFrame of weighted game box score stats for one player.
	"""
	if "seconds_played_y" in weighted_stats.index:
		return np.mean(weighted_stats["seconds_played_y"])/60
	else:
		return np.mean(weighted_stats["seconds_played"])/60

def recent_average(weighted_stats):
	"""Produces an average of recent Fanduel points for the weighted statlines. Because the statlines are already weighted
	versions of the past several games, this is not giving the actual recent average of it but a recent average
	that is weighted more closely to recent performances.
	
	Params:
	weighted_stats: A pandas DataFrame of weighted game box score stats for one player.
	"""
	if len(weighted_stats.index) == 0:
		return 0
	if "made_three_point_field_goals_y" in weighted_stats.index:
		return 3*np.mean(weighted_stats["made_three_point_field_goals_y"]) + 2*np.mean(weighted_stats["made_two_point_field_goals_y"]) + np.mean(weighted_stats["made_free_throws_y"]) + 1.2*(np.mean(weighted_stats["offensive_rebounds_y"]) + np.mean(weighted_stats["defensive_rebounds_y"])) + 1.5*np.mean(weighted_stats["assists_y"]) + 3*np.mean(weighted_stats["blocks_y"]) + 3*np.mean(weighted_stats["steals_y"]) - np.mean(weighted_stats["turnovers_y"])
	else:
		return 3*np.mean(weighted_stats["made_three_point_field_goals"]) + 2*np.mean(weighted_stats["made_two_point_field_goals"]) + np.mean(weighted_stats["made_free_throws"]) + 1.2*(np.mean(weighted_stats["offensive_rebounds"]) + np.mean(weighted_stats["defensive_rebounds"])) + 1.5*np.mean(weighted_stats["assists"]) + 3*np.mean(weighted_stats["blocks"]) + 3*np.mean(weighted_stats["steals"]) - np.mean(weighted_stats["turnovers"])


def matchup_lookup(matchups, team):
	"""Returns, given a schedule of games, the opponent of a team playing in the schedule.

	Params:
	matchups: A list of tuples where each tuple is a game taking place in the schedule.
	teams: A string representing a team competing in one of the games on the schedule.
	"""
	matchup = [m for m in matchups if team in m][0]
	if team == matchup[0]:
		return matchup[1]
	else:
		return matchup[0]

def box_scores_for_range_of_days(start_date, end_date):
	"""Returns pandas DataFrame of all basketball-reference box score statlines between start_date and end_date.
	
	Params:
	start_date: Datetime object representing date to begin collecting data from.
	end_date. Datetime object representing last date to collect data for.
	"""
	all_tables = []
	start_month = start_date.month
	end_month = end_date.month
	start_year = start_date.year
	end_year = end_date.year
	
	for y in range(start_year, end_year + 1):
		sm, em = 1, 12
		if y == start_year:
			sm = start_month
		if y == end_year:
			em = end_month
		for m in range(sm, em + 1):
			if m == start_month and y == start_year:
				start_day = start_date.day
			else:
				start_day = 1
			if m == end_month and y == end_year:
				end_day = end_date.day
			else:
				if m == 2:
					if y % 4 == 0:
						end_day = 29
					else:
						end_day = 28
				elif m in [9, 4, 6, 11]:
					end_day = 30
				else:
					end_day = 31
		
			for d in range(start_day, end_day + 1):
				file_name = "./AllCSVs/{0}_{1}_{2}_box_scores.csv".format(m, d, y)
				if not path.exists(file_name):
					continue
				if pd.read_csv(file_name).empty:
					client.player_box_scores(day=d, month=m, year=y, output_type=OutputType.CSV, output_file_path=file_name)
				table = pd.read_csv(file_name)
				date = datetime(y, m, d)
				dates = [date] * len(table)
				table["Date"] = dates
				all_tables.append(table)

	full_df = all_tables[0]
	for i in range(1, len(all_tables)):
		current_table = all_tables[i]
		full_df = full_df.append(current_table)
	
	full_df.index = range(full_df.shape[0])
	df = pd.read_csv("./OutputCSVs/updated_team_stats.csv")
	df["team"] = df["team"].str.upper()

	team_def = []
	team_pace = []
	team_tov = []
	opp_def = []
	opp_pace = []
	opp_tov = []
	all_games_teams = full_df[["team", "opponent"]]

	for i in range(len(all_games_teams)):
		game = all_games_teams.loc[i]
		team = game["team"].upper()
		opponent = game["opponent"].upper()
		team_def.append(df[df["team"] == team]["drtg"].iloc[0])
		team_pace.append(df[df["team"] == team]["pace"].iloc[0])
		team_tov.append(df[df["team"] == team]["tov%"].iloc[0])
		opp_def.append(df[df["team"] == opponent]["drtg"].iloc[0])
		opp_pace.append(df[df["team"] == opponent]["pace"].iloc[0])
		opp_tov.append(df[df["team"] == opponent]["tov%"].iloc[0])
	full_df["Team Defensive Rating"] = team_def
	full_df["Team Pace"] = team_pace
	full_df["Team Turnover %"] = team_tov
	full_df["Opponent Defensive Rating"] = opp_def
	full_df["Opponent Pace"] = opp_pace
	full_df["Opponent Turnover %"] = opp_tov
	return full_df