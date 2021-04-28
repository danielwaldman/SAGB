from hardcoded import *
from modeling_helpers import *
from processing import *

from basketball_reference_web_scraper import client
import pandas as pd
import datetime as datetime
from datetime import datetime, timedelta
from pulp import LpVariable, LpProblem, LpMaximize
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
import os.path
from os import path
import joblib
import tempfile


def statline_output(player_box_scores, input_statistics, per_min = False, retrain = True, pretrain_inputs = True):
	"""Here is the meat of the modeling process. Takes in player_box_scores, a set of box scores from the
	desired start date of the data until the game date, and produces several versions of weighted statlines
	using generate_input_vector and consisting of the statistics in input_statistics. We then keep the weighted statlines
	with over 800 seconds played (potential yikes here) and use these as model inputs in concurrent models
	for each of the statistics in output_statistics. Each of the output_statistics has a corresponding model
	and best set of weighted lines which was determined through testing. We keep the data for model inputs, train
	the models by merging the weighted lines with the actual lines and predicting each actual statistic with the weighted
	lines, and then make an empty DataFrame that we fill with predictions using these models. This DataFrame is processed
	and returned.
	
	Params:
	player_box_scores: Box scores over the desired range of days
	input_statistics: Array of statistics to be used as model inputs.
	
	TODO:
		-Make this predict FPPM. Generally make the prediction process more intelligent.
	"""
	input_indices = [3, 7, 6, 9, 10, 12, 11, 13, 14, 15, 18, 16, 17, 22, 23, 24, 20, 25, 21, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
	output_indices = {"seconds": 51,
					  "threepoints": 54,
					  "freethrows": 56,
					  "assists": 60,
					  "steals": 61,
					  "blocks": 62,
					  "turnovers": 63,
					  "twopoints": 74,
					  "rebounds":75}
	
	if not pretrain_inputs:
		input_indices = [i + 1 for i in input_indices]
		for stat in output_indices.keys():
			output_indices[stat] = output_indices[stat] + 1
		
	
	if per_min:
		for box_index in player_box_scores.index:
			mins = player_box_scores.loc[box_index, "seconds_played"]
			for j in player_box_scores.columns[6:20]:
				if mins > 0:
					player_box_scores.loc[box_index, j] = player_box_scores.loc[box_index, j]*60/player_box_scores.loc[box_index, "seconds_played"]
	
	# Creating the four sets of weighted trailing statlines, using sample sizes of 7 and 8
	# and weights of .8, .85 and .9.
	
	
	filename_685 = "./pretrained/685.csv"
	filename_985 = "./pretrained/985.csv"
	filename_1085 = "./pretrained/1085.csv"
	
	if path.exists(filename_685):
		if not pretrain_inputs:
			weighted_lines_6_85 = pd.read_csv(filename_685)
			print("a")
		else:
			weighted_lines_6_85 = generate_input_vector(player_box_scores, input_statistics, 6, .85, per_min)
			weighted_lines_6_85.to_csv(filename_685)
			print("b")
	else:
		weighted_lines_6_85 = generate_input_vector(player_box_scores, input_statistics, 6, .85, per_min)
		weighted_lines_6_85.to_csv(filename_685)
		print("c")
	if path.exists(filename_985):
		if not pretrain_inputs:
			weighted_lines_9_85 = pd.read_csv(filename_985)
		else:
			weighted_lines_9_85 = generate_input_vector(player_box_scores, input_statistics, 9, .85, per_min) 
			weighted_lines_9_85.to_csv(filename_985)
	else:
		weighted_lines_9_85 = generate_input_vector(player_box_scores, input_statistics, 9, .85, per_min) 
		weighted_lines_9_85.to_csv(filename_985)
	if path.exists(filename_1085):
		if not pretrain_inputs:
			weighted_lines_10_85 = pd.read_csv(filename_1085)
		else:
			weighted_lines_10_85 = generate_input_vector(player_box_scores, input_statistics, 10, .85, per_min)
			weighted_lines_10_85.to_csv(filename_1085)
	else:
		weighted_lines_10_85 = generate_input_vector(player_box_scores, input_statistics, 10, .85, per_min)
		weighted_lines_10_85.to_csv(filename_1085)
				
	weighted_lines_to_keep_6_85 = weighted_lines_6_85
	weighted_lines_to_keep_9_85 = weighted_lines_9_85
	weighted_lines_to_keep_10_85 = weighted_lines_10_85
	player_box_scores["Date"] = player_box_scores["Date"].astype(str)
	df_to_keep = player_box_scores[~player_box_scores["Date"].str.contains("2020-10")]
	
	print(weighted_lines_to_keep_6_85)
	
	# Doing some processing to merge name and date and ensure we can access unique statlines.
	
	df_to_keep["attempted_two_point_field_goals"] = df_to_keep["attempted_field_goals"] - df_to_keep["attempted_three_point_field_goals"]
	df_to_keep["made_two_point_field_goals"] = df_to_keep["made_field_goals"] - df_to_keep["made_three_point_field_goals"]
	weighted_lines_to_keep_6_85['name_date'] = weighted_lines_to_keep_6_85["name"] + weighted_lines_to_keep_6_85["date"].astype(str)
	weighted_lines_to_keep_9_85['name_date'] = weighted_lines_to_keep_9_85["name"] + weighted_lines_to_keep_9_85["date"].astype(str)
	weighted_lines_to_keep_10_85['name_date'] = weighted_lines_to_keep_10_85["name"] + weighted_lines_to_keep_10_85["date"].astype(str)
	df_to_keep['name_date'] = df_to_keep["name"] + df_to_keep["Date"]
	
	# Merging the weighted lines (which are just weighted versions of past performances) with the actual lines.
	# We train the models using the weighted lines as inputs to predict each statistic in actual_lines,
	# which means merging both into the same DataFrame. A little processing is required to get the columns
	# to match with output_statistics.
	def merge_with_actual(data, actual):
		merged = data.merge(actual, left_on = 'name_date', right_on = 'name_date')
		merged["rebounds_y"] = merged["offensive_rebounds_y"] + merged["defensive_rebounds_y"]
		merged["location_x"] = merged["location_x"] == "HOME"
		merged["location_y"] = merged["location_y"] == "HOME"
		print(merged.shape, data.shape, actual.shape)
		return merged
		
	df_merged_6_85 = merge_with_actual(weighted_lines_to_keep_6_85, df_to_keep)
	df_merged_9_85 = merge_with_actual(weighted_lines_to_keep_9_85, df_to_keep)
	df_merged_10_85 = merge_with_actual(weighted_lines_to_keep_10_85, df_to_keep)
	
	# Here we isolate the statistics from the weighted lines that we will use as model inputs.
	
	print(len(df_merged_6_85.columns))
	print(df_merged_6_85.columns)
	
	predictors_6_85 = df_merged_6_85.iloc[:, input_indices]
	predictors_9_85 = df_merged_9_85.iloc[:, input_indices]
	predictors_10_85 = df_merged_10_85.iloc[:, input_indices]
	
	print(predictors_6_85)
	
	# Producing the number of fantasy points for each game. Not really relevant but we used it to see if we could
	# make fantasy_points an output to predict with the weighted lines. Turned out not to be very useful, but worth keeping
	# around to see if future models have a better time with it.
	
	df_merged_6_85["fantasy_points"] = [float(get_points(df_merged_6_85[df_merged_6_85["name_date"] == player_name])[0]) for player_name in df_merged_6_85["name_date"]]
	df_merged_9_85["fantasy_points"] = [float(get_points(df_merged_9_85[df_merged_9_85["name_date"] == player_name])[0]) for player_name in df_merged_9_85["name_date"]]
	df_merged_10_85["fantasy_points"] = [float(get_points(df_merged_10_85[df_merged_10_85["name_date"] == player_name])[0]) for player_name in df_merged_10_85["name_date"]]
	
	# Getting the target output columns from their respective weighted line DataFrames.
	#fantasy_points_8_9 = df_merged_8_9.iloc[:,60]
	seconds = df_merged_6_85.iloc[:, output_indices["seconds"]]
	seconds = seconds.astype(float)
	threepoints = df_merged_10_85.iloc[:, output_indices["threepoints"]]
	freethrows = df_merged_10_85.iloc[:, output_indices["freethrows"]]
	assists = df_merged_10_85.iloc[:, output_indices["assists"]]
	steals = df_merged_9_85.iloc[:, output_indices["steals"]]
	blocks = df_merged_10_85.iloc[:, output_indices["blocks"]]
	turnovers = df_merged_9_85.iloc[:, output_indices["turnovers"]]
	twopoints = df_merged_10_85.iloc[:, output_indices["twopoints"]]
	rebounds = df_merged_10_85.iloc[:, output_indices["rebounds"]]

	
	# Training the models! Using the predictors and their corresponding output, selected from the weighted training
	# data that we tested to predict the outputs the best.
	
	def get_model(stat, model, predictors, target, retrain = False):
		fileext = stat + ".joblib"
		temp = tempfile.mkdtemp()
		filename = os.path.join(temp, fileext)
		if not path.exists(fileext) or retrain:
			output = model.fit(predictors, target)
			joblib.dump(output, filename)
			return output
		else:
			return joblib.load(filename, fileext)

	freethrow_model = get_model("freethrow", RidgeCV(), predictors_10_85, freethrows, retrain)
	twopoint_model = get_model("twopoint", BayesianRidge(n_iter = 400), predictors_10_85, twopoints, retrain)
	threepoint_model = get_model("threepoint", GradientBoostingRegressor(), predictors_10_85, threepoints, retrain)
	block_model = get_model("block", RidgeCV(), predictors_10_85, blocks, retrain)
	assist_model = get_model("assist", BayesianRidge(n_iter = 400), predictors_10_85, assists, retrain)
	rebound_model = get_model("rebound", RidgeCV(), predictors_10_85, rebounds, retrain)
	turnover_model = get_model("turnover", RidgeCV(), predictors_9_85, turnovers, retrain)
	steal_model = get_model("steal", RidgeCV(), predictors_9_85, steals, retrain)
	second_model = get_model("second", GradientBoostingRegressor(), predictors_6_85, seconds, retrain)

	# Creating blank output statlines and matching the non-numeric details.
	
	output_statlines = pd.DataFrame(index = weighted_lines_10_85.index, columns = output_statistics).fillna(0)
	output_statlines["name"] = weighted_lines_10_85["name"]
	output_statlines["team"] = weighted_lines_10_85["team"]
	output_statlines["date"] = weighted_lines_10_85["date"]
	output_statlines["location"] = weighted_lines_10_85["location"]
	output_statlines["opponent"] = weighted_lines_10_85["opponent"]
	
	# Processing the weighted lines for use as predictive inputs in the trained models.

	weighted_lines_6_85["location"] = weighted_lines_6_85["location"] == "HOME"
	weighted_lines_9_85["location"] = weighted_lines_9_85["location"] == "HOME"
	weighted_lines_10_85["location"] = weighted_lines_10_85["location"] == "HOME"
	weighted_lines_6_85 = weighted_lines_6_85.iloc[:, input_indices]
	weighted_lines_9_85 = weighted_lines_9_85.iloc[:, input_indices]
	weighted_lines_10_85 = weighted_lines_10_85.iloc[:, input_indices]
	#weighted_lines_8_9 = weighted_lines_8_9.rename(columns={'made_two_point_field_goals': 'made_two_point_field_goals_x', 'attempted_two_point_field_goals': 'attempted_two_point_field_goals_x'})
	
	# Using the models to predict each statistic! We then fill in output_statlines with each prediction.
	
	output_statlines["minutes"] = second_model.predict(weighted_lines_6_85) / 60
	output_statlines["made_two_point_field_goals"] = twopoint_model.predict(weighted_lines_10_85)
	output_statlines["made_three_point_field_goals"] = threepoint_model.predict(weighted_lines_10_85) 
	output_statlines["made_free_throws"] = freethrow_model.predict(weighted_lines_10_85)
	output_statlines["rebounds"] = rebound_model.predict(weighted_lines_10_85)
	output_statlines["assists"] = assist_model.predict(weighted_lines_10_85)
	output_statlines["blocks"] = block_model.predict(weighted_lines_10_85)
	output_statlines["steals"] = steal_model.predict(weighted_lines_9_85) 
	output_statlines["turnovers"] = turnover_model.predict(weighted_lines_9_85)
	#output_statlines["fantasy_points_8_9"] = fantasy_model_8_9.predict(weighted_lines_8_9)
	
	# Going through each row and making all the outputs clean. Dealing with weird outliers and edge cases.
	
	for box_index in output_statlines.index:
#         if output_statlines.loc[box_index, "fantasy_points_8_9"] < -100:
#             output_statlines.loc[box_index, "minutes"] = 1
		pred_minutes = max(0, output_statlines.loc[box_index, "minutes"])
		if pred_minutes <= 19:
			output_statlines.loc[box_index, "made_two_point_field_goals"] = output_statlines.loc[box_index, "made_two_point_field_goals"] * pred_minutes/19
			output_statlines.loc[box_index, "made_three_point_field_goals"] = output_statlines.loc[box_index, "made_three_point_field_goals"] * pred_minutes/19
			output_statlines.loc[box_index, "made_free_throws"] = output_statlines.loc[box_index, "made_free_throws"] * pred_minutes/19
			output_statlines.loc[box_index, "rebounds"] = output_statlines.loc[box_index, "rebounds"] * pred_minutes/19
			output_statlines.loc[box_index, "assists"] = output_statlines.loc[box_index, "assists"] * pred_minutes/19
			output_statlines.loc[box_index, "blocks"] = output_statlines.loc[box_index, "blocks"] * pred_minutes/19
			output_statlines.loc[box_index, "steals"] = output_statlines.loc[box_index, "steals"] * pred_minutes/19
			output_statlines.loc[box_index, "turnovers"] = output_statlines.loc[box_index, "turnovers"] * pred_minutes/19
		output_statlines.loc[box_index, "minutes"] = round(pred_minutes, 2)
		output_statlines.loc[box_index, "made_two_point_field_goals"] = round(max(0, output_statlines.loc[box_index, "made_two_point_field_goals"]), 2)
		output_statlines.loc[box_index, "made_three_point_field_goals"] = round(max(0, output_statlines.loc[box_index, "made_three_point_field_goals"]), 2)
		output_statlines.loc[box_index, "made_free_throws"] = round(max(0, output_statlines.loc[box_index, "made_free_throws"]), 2)
		output_statlines.loc[box_index, "rebounds"] = round(max(0, output_statlines.loc[box_index, "rebounds"]), 2)
		output_statlines.loc[box_index, "assists"] = round(max(0, output_statlines.loc[box_index, "assists"]), 2)
		output_statlines.loc[box_index, "blocks"] = round(max(0, output_statlines.loc[box_index, "blocks"]), 2)
		output_statlines.loc[box_index, "steals"] = round(max(0, output_statlines.loc[box_index, "steals"]), 2)
		output_statlines.loc[box_index, "turnovers"] = round(max(0, output_statlines.loc[box_index, "turnovers"]), 2)
		output_statlines.loc[box_index, "recent_average"] = round(np.mean([recent_average(weighted_lines_10_85.loc[box_index]), recent_average(weighted_lines_6_85.loc[box_index])]), 2)
		last_10_games = get_stats(output_statlines.loc[box_index, "name"], output_statlines.loc[box_index, "date"], 10)
		last_3_games = get_stats(output_statlines.loc[box_index, "name"], output_statlines.loc[box_index, "date"], 3)
		output_statlines.loc[box_index, "10_game_average"] = round(recent_average(last_10_games), 2)
		output_statlines.loc[box_index, "3_game_average"] = round(recent_average(last_3_games), 2)
		output_statlines.loc[box_index, "10_3_ratio"] = (output_statlines.loc[box_index, "10_game_average"] + 1)/(output_statlines.loc[box_index, "3_game_average"] + 1)
		output_statlines.loc[box_index, "10_3_difference"] = output_statlines.loc[box_index, "10_game_average"] - output_statlines.loc[box_index, "3_game_average"]
		output_statlines.loc[box_index, "hot"] = np.log(((-7 * min(0, output_statlines.loc[box_index, "10_3_ratio"] - .83)) * (-1 * min(0, output_statlines.loc[box_index, "10_3_difference"] + 6))) + 1)
		output_statlines.loc[box_index, "cold"] = np.log(((7 * max(0, output_statlines.loc[box_index, "10_3_ratio"] - 1.22)) * (max(0, output_statlines.loc[box_index, "10_3_difference"] - 6.5))) + 1)
	
	# Adding team defense onto the statlines.
	
	output_statlines = add_team_defense(output_statlines)
	return output_statlines


def regression_by_sample(start_date, end_date, sample_size, weight, per_min = False):
	"""Fits models to random samples of training data on each of the output statistics.
	This function is useful for model selection, validation and testing, and for tuning the sample_size, weight
	and min_seconds hyperparameters. Fits several models to each output statistic using multiple different
	random train-test splits and displays model accuracy in r^2 score and MSE.
	
	Params:
	start_date: Datetime object representing date to begin collecting data from.
	end_date: Datetime object representing date to collect data until.
	sample_size: Integer representing length of sample window to produce weighted average statlines with.
	weight: Float representing weight to bias weighted average towards recent results.
	per_min: Boolean representing whether we are doing per minute regression.
	
	"""
	
	# Here we follow the same procedure as in statline_output to generate inputs and outputs
	# suitable for modeling and prediction.
	
	full_df = box_scores_for_range_of_days(start_date, end_date)
	
	input_indices = [3, 7, 6, 9, 10, 12, 11, 13, 14, 15, 18, 16, 17, 22, 23, 24, 20, 25, 21, 26, 27]
	
	weighted_statlines = generate_input_vector(full_df, input_statistics, sample_size, weight)
	
	
	if per_min:
		for box_index in full_df.index:
			mins = full_df.loc[box_index, "seconds_played"]
			for j in full_df.columns[7:21]:
				if mins > 0:
					full_df.loc[box_index, j] = full_df.loc[box_index, j]*60/full_df.loc[box_index, "seconds_played"]
					
	print(full_df.head())
	print(full_df.columns)
	
	#weighted_statlines_by_min = generate_input_vector(full_df, input_statistics, sample_size, weight)
	weighted_statlines_to_keep = weighted_statlines
	df_to_keep = full_df
	df_to_keep["attempted_two_point_field_goals"] = df_to_keep["attempted_field_goals"] - df_to_keep["attempted_three_point_field_goals"]
	df_to_keep["made_two_point_field_goals"] = df_to_keep["made_field_goals"] - df_to_keep["made_three_point_field_goals"]
	weighted_statlines_to_keep['name_date'] = weighted_statlines_to_keep["name"] + weighted_statlines_to_keep["date"].astype(str)
	df_to_keep['name_date'] = df_to_keep["name"] + df_to_keep["Date"].astype(str)
	df_merged = weighted_statlines_to_keep.merge(df_to_keep, left_on = 'name_date', right_on = 'name_date')
	df_merged["rebounds_y"] = df_merged["offensive_rebounds_y"] + df_merged["defensive_rebounds_y"]
	df_merged["location_x"] = df_merged["location_x"] == "HOME"
	df_merged["location_y"] = df_merged["location_y"] == "HOME"
	print(df_merged.columns)
	df_merged["fantasy_points"] = [float(get_points(df_merged[df_merged["name_date"] == player_name])[0]) for player_name in df_merged["name_date"]]
	df_merged = df_merged[df_merged.Date.astype(str).str.contains("2020-02")]
	predictors = df_merged.iloc[:, input_indices]
	print(df_merged.columns[input_indices])
	print(df_merged.columns[~input_indices])
	print(len(df_merged.columns))
	print(df_merged.head())
	
	
	# Right now this just models seconds played. To get this to model the other output
	# statistics you can uncomment the rest of the indices in the list.
	
	for desired_output in [35, 38, 40, 44, 45, 46, 47, 48, 58, 59]:
		colname = df_merged.columns[desired_output]
		print(colname)
		y = df_merged.iloc[:,desired_output]
		
		pred_train, pred_test, y_train, y_test = train_test_split(predictors, y, test_size=0.1, random_state=85733)
		pred_train1, pred_test1, y_train1, y_test1 = train_test_split(predictors, y, test_size=0.1, random_state=433)
		pred_train2, pred_test2, y_train2, y_test2 = train_test_split(predictors, y, test_size=0.1, random_state=96323)
		pred_train3, pred_test3, y_train3, y_test3 = train_test_split(predictors, y, test_size=0.1, random_state=76243)
		pred_train4, pred_test4, y_train4, y_test4 = train_test_split(predictors, y, test_size=0.1, random_state=76343)
#         x_train, x_test = xgb.DMatrix(pred_train, label = y_train, enable_categorical= True), xgb.DMatrix(pred_test, label = y_test, enable_categorical= True)
#         x_train1, x_test1 = xgb.DMatrix(pred_train1, label = y_train1, enable_categorical = True), xgb.DMatrix(pred_test1, label = y_test1, enable_categorical= True)
#         x_train2, x_test2 = xgb.DMatrix(pred_train2, label = y_train2, enable_categorical = True), xgb.DMatrix(pred_test2, label = y_test2, enable_categorical= True)
#         x_train3, x_test3 = xgb.DMatrix(pred_train3, label = y_train3, enable_categorical = True), xgb.DMatrix(pred_test3, label = y_test3, enable_categorical= True)
#         x_train4, x_test4 = xgb.DMatrix(pred_train4, label = y_train4, enable_categorical = True), xgb.DMatrix(pred_test4, label = y_test4, enable_categorical= True)
#         x_param = {'eta': 0.25, 'max_depth': 5, 'objective': 'reg:gamma', 'eval_metric': 'mae'}
		
		y_train = y_train.astype(float)
		gaussian_model = GaussianProcessRegressor().fit(pred_train, y_train)
		decisiontree_model = DecisionTreeRegressor().fit(pred_train, y_train)
		sgd_model = SGDRegressor(loss = 'huber').fit(pred_train, y_train)
		ridge_model = RidgeCV().fit(pred_train, y_train)
#         x_model = xgb.train(x_param, x_train, 25)
		ard_model = ARDRegression(n_iter = 400).fit(pred_train, y_train)
		bayesian_model = BayesianRidge(n_iter = 400).fit(pred_train, y_train)
		neural_model = MLPRegressor(max_iter = 600).fit(pred_train, y_train)
		cat_model = CatBoostRegressor().fit(pred_train, y_train, cat_features = ["location_x"], verbose_eval = False)
		linear_model = GradientBoostingRegressor().fit(pred_train, y_train)
		print("fit 0")
		ridge_model1 = RidgeCV().fit(pred_train1, y_train1)
		y_train1 = y_train1.astype(float)
		gaussian_model1 = GaussianProcessRegressor().fit(pred_train1, y_train1)
		decisiontree_model1 = DecisionTreeRegressor().fit(pred_train1, y_train1)
		sgd_model1 = SGDRegressor(loss = 'huber').fit(pred_train1, y_train1)
#         x_model1 = xgb.train(x_param, x_train1, 25)
		ard_model1 = ARDRegression(n_iter = 400).fit(pred_train1, y_train1)
		bayesian_model1 = BayesianRidge(n_iter = 400).fit(pred_train1, y_train1)
		neural_model1 = MLPRegressor(max_iter = 600).fit(pred_train1, y_train1)
		cat_model1 = CatBoostRegressor().fit(pred_train1, y_train1, cat_features = ["location_x"], verbose_eval = False)
		linear_model1 = GradientBoostingRegressor().fit(pred_train1, y_train1)
		print("fit 1")
		ridge_model2 = RidgeCV().fit(pred_train2, y_train2)
		y_train2 = y_train2.astype(float)
		gaussian_model2 = GaussianProcessRegressor().fit(pred_train2, y_train2)
		decisiontree_model2 = DecisionTreeRegressor().fit(pred_train2, y_train2)
		sgd_model2 = SGDRegressor(loss = 'huber').fit(pred_train2, y_train2)
#         x_model2 = xgb.train(x_param, x_train2, 25)
		ard_model2 = ARDRegression(n_iter = 400).fit(pred_train2, y_train2)
		bayesian_model2 = BayesianRidge(n_iter = 400).fit(pred_train2, y_train2)
		neural_model2 = MLPRegressor(max_iter = 600).fit(pred_train2, y_train2)
		cat_model2 = CatBoostRegressor().fit(pred_train2, y_train2, cat_features = ["location_x"], verbose_eval = False)
		linear_model2 = GradientBoostingRegressor().fit(pred_train2, y_train2)
		print("fit 2")
		ridge_model3 = RidgeCV().fit(pred_train3, y_train3)
		y_train3 = y_train3.astype(float)
		gaussian_model3 = GaussianProcessRegressor().fit(pred_train3, y_train3)
		decisiontree_model3 = DecisionTreeRegressor().fit(pred_train3, y_train3)
		sgd_model3 = SGDRegressor(loss = 'huber').fit(pred_train3, y_train3)
#         x_model3 = xgb.train(x_param, x_train3, 25)
		ard_model3 = ARDRegression(n_iter = 400).fit(pred_train3, y_train3)
		bayesian_model3 = BayesianRidge(n_iter = 400).fit(pred_train3, y_train3)
		neural_model3 = MLPRegressor(max_iter = 600).fit(pred_train3, y_train3)
		cat_model3 = CatBoostRegressor().fit(pred_train3, y_train3, cat_features = ["location_x"], verbose_eval = False)
		linear_model3 = GradientBoostingRegressor().fit(pred_train3, y_train3)
		print("fit 3")
		ridge_model4 = RidgeCV().fit(pred_train4, y_train4)
		y_train4 = y_train4.astype(float)
		gaussian_model4 = GaussianProcessRegressor().fit(pred_train4, y_train4)
		decisiontree_model4 = DecisionTreeRegressor().fit(pred_train4, y_train4)
		sgd_model4 = SGDRegressor(loss = 'huber').fit(pred_train4, y_train4)
		ard_model4 = ARDRegression(n_iter = 400).fit(pred_train4, y_train4)
		bayesian_model4 = BayesianRidge(n_iter = 400).fit(pred_train4, y_train4)
		neural_model4 = MLPRegressor(max_iter = 600).fit(pred_train4, y_train4)
		cat_model4 = CatBoostRegressor().fit(pred_train4, y_train4, cat_features = ["location_x"], verbose_eval = False)
		linear_model4 = GradientBoostingRegressor().fit(pred_train4, y_train4)
		print("fit 4")
#         ridge_model5 = RidgeCV().fit(pred_train5, y_train5)
#         y_train5 = y_train5.astype(float)
#         cat_model5 = CatBoostRegressor().fit(pred_train5, y_train5, cat_features = ["location_x"], verbose_eval = False)
#         linear_model5 = GradientBoostingRegressor().fit(pred_train5, y_train5)
#         ridge_model6 = RidgeCV().fit(pred_train6, y_train6)
#         y_train6 = y_train6.astype(float)
#         cat_model6 = CatBoostRegressor().fit(pred_train6, y_train6, cat_features = ["location_x"], verbose_eval = False)
#         linear_model6 = GradientBoostingRegressor().fit(pred_train6, y_train6)
#         print(df_merged.columns[desired_output])
		
		
		y_test = y_test.astype(float)
		y_test1 = y_test1.astype(float)
		y_test2 = y_test2.astype(float)
		y_test3 = y_test3.astype(float)
		y_test4 = y_test4.astype(float)
#         y_test5 = y_test5.astype(float)
#         y_test6 = y_test6.astype(float)
		
		results_df = pd.DataFrame(columns=["Gaussian", "Decision Tree", "SGD", "RidgeCV", "ARD", "Bayesian", "Neural", "Cat", "Linear"])
		errors_model0 = [np.mean(abs(gaussian_model.predict(pred_test) - y_test)),
			np.mean(abs(decisiontree_model.predict(pred_test) - y_test)),
			np.mean(abs(sgd_model.predict(pred_test) - y_test)),
			np.mean(abs(ridge_model.predict(pred_test) - y_test)),
			np.mean(abs(ard_model.predict(pred_test) - y_test)),
			np.mean(abs(bayesian_model.predict(pred_test) - y_test)),
			np.mean(abs(neural_model.predict(pred_test) - y_test)),
			np.mean(abs(cat_model.predict(pred_test) - y_test)),
			np.mean(abs(linear_model.predict(pred_test) - y_test))]
		scores_model0 = [gaussian_model.score(pred_test, y_test),
			decisiontree_model.score(pred_test, y_test),
			sgd_model.score(pred_test, y_test),
			ridge_model.score(pred_test, y_test),
			ard_model.score(pred_test, y_test),
			bayesian_model.score(pred_test, y_test),
			neural_model.score(pred_test, y_test),
			cat_model.score(pred_test, y_test),
			linear_model.score(pred_test, y_test)]
		errors_model1 = [np.mean(abs(gaussian_model1.predict(pred_test1) - y_test1)),
			np.mean(abs(decisiontree_model1.predict(pred_test1) - y_test1)),
			np.mean(abs(sgd_model1.predict(pred_test1) - y_test1)),
			np.mean(abs(ridge_model1.predict(pred_test1) - y_test1)),
			np.mean(abs(ard_model1.predict(pred_test1) - y_test1)),
			np.mean(abs(bayesian_model1.predict(pred_test1) - y_test1)),
			np.mean(abs(neural_model1.predict(pred_test1) - y_test1)),
			np.mean(abs(cat_model1.predict(pred_test1) - y_test1)),
			np.mean(abs(linear_model1.predict(pred_test1) - y_test1))]
		scores_model1 = [gaussian_model1.score(pred_test1, y_test1),
			decisiontree_model1.score(pred_test1, y_test1),
			sgd_model1.score(pred_test1, y_test1),
			ridge_model1.score(pred_test1, y_test1),
			ard_model1.score(pred_test1, y_test1),
			bayesian_model1.score(pred_test1, y_test1),
			neural_model1.score(pred_test1, y_test1),
			cat_model1.score(pred_test1, y_test1),
			linear_model1.score(pred_test1, y_test1)]
		errors_model2 = [np.mean(abs(gaussian_model2.predict(pred_test2) - y_test2)),
			np.mean(abs(decisiontree_model2.predict(pred_test2) - y_test2)),
			np.mean(abs(sgd_model2.predict(pred_test2) - y_test2)),
			np.mean(abs(ridge_model2.predict(pred_test2) - y_test2)),
			np.mean(abs(ard_model2.predict(pred_test2) - y_test2)),
			np.mean(abs(bayesian_model2.predict(pred_test2) - y_test2)),
			np.mean(abs(neural_model2.predict(pred_test2) - y_test2)),
			np.mean(abs(cat_model2.predict(pred_test2) - y_test2)),
			np.mean(abs(linear_model2.predict(pred_test2) - y_test2))]
		scores_model2 = [gaussian_model2.score(pred_test2, y_test2),
			decisiontree_model2.score(pred_test2, y_test2),
			sgd_model2.score(pred_test2, y_test2),
			ridge_model2.score(pred_test2, y_test2),
			ard_model2.score(pred_test2, y_test2),
			bayesian_model2.score(pred_test2, y_test2),
			neural_model2.score(pred_test2, y_test2),
			cat_model2.score(pred_test2, y_test2),
			linear_model2.score(pred_test2, y_test2)]
		errors_model3 = [np.mean(abs(gaussian_model3.predict(pred_test3) - y_test3)),
			np.mean(abs(decisiontree_model3.predict(pred_test3) - y_test3)),
			np.mean(abs(sgd_model3.predict(pred_test3) - y_test3)),
			np.mean(abs(ridge_model3.predict(pred_test3) - y_test3)),
			np.mean(abs(ard_model3.predict(pred_test3) - y_test3)),
			np.mean(abs(bayesian_model3.predict(pred_test3) - y_test3)),
			np.mean(abs(neural_model3.predict(pred_test3) - y_test3)),
			np.mean(abs(cat_model3.predict(pred_test3) - y_test3)),
			np.mean(abs(linear_model3.predict(pred_test3) - y_test3))]
		scores_model3 = [gaussian_model3.score(pred_test3, y_test3),
			decisiontree_model3.score(pred_test3, y_test3),
			sgd_model3.score(pred_test3, y_test3),
			ridge_model3.score(pred_test3, y_test3),
			ard_model3.score(pred_test3, y_test3),
			bayesian_model3.score(pred_test3, y_test3),
			neural_model3.score(pred_test3, y_test3),
			cat_model3.score(pred_test3, y_test3),
			linear_model3.score(pred_test3, y_test3)]
		errors_model4 = [np.mean(abs(gaussian_model4.predict(pred_test4) - y_test4)),
			np.mean(abs(decisiontree_model4.predict(pred_test4) - y_test4)),
			np.mean(abs(sgd_model4.predict(pred_test4) - y_test4)),
			np.mean(abs(ridge_model4.predict(pred_test4) - y_test4)),
			np.mean(abs(ard_model4.predict(pred_test4) - y_test4)),
			np.mean(abs(bayesian_model4.predict(pred_test4) - y_test4)),
			np.mean(abs(neural_model4.predict(pred_test4) - y_test4)),
			np.mean(abs(cat_model4.predict(pred_test4) - y_test4)),
			np.mean(abs(linear_model4.predict(pred_test4) - y_test4))]
		scores_model4 = [gaussian_model4.score(pred_test4, y_test4),
			decisiontree_model4.score(pred_test4, y_test4),
			sgd_model4.score(pred_test4, y_test4),
			ridge_model4.score(pred_test4, y_test4),
			ard_model4.score(pred_test4, y_test4),
			bayesian_model4.score(pred_test4, y_test4),
			neural_model4.score(pred_test4, y_test4),
			cat_model4.score(pred_test4, y_test4),
			linear_model4.score(pred_test4, y_test4)]
		
		results_df = results_df.append(pd.Series(errors_model0, index = results_df.columns), ignore_index = True)
		results_df = results_df.append(pd.Series(scores_model0, index = results_df.columns), ignore_index = True)
		results_df = results_df.append(pd.Series(errors_model1, index = results_df.columns), ignore_index = True)
		results_df = results_df.append(pd.Series(scores_model1, index = results_df.columns), ignore_index = True)
		results_df = results_df.append(pd.Series(errors_model2, index = results_df.columns), ignore_index = True)
		results_df = results_df.append(pd.Series(scores_model2, index = results_df.columns), ignore_index = True)
		results_df = results_df.append(pd.Series(errors_model3, index = results_df.columns), ignore_index = True)
		results_df = results_df.append(pd.Series(scores_model3, index = results_df.columns), ignore_index = True)
		results_df = results_df.append(pd.Series(errors_model4, index = results_df.columns), ignore_index = True)
		results_df = results_df.append(pd.Series(scores_model4, index = results_df.columns), ignore_index = True)
		

#         print(np.mean((ridge_model5.predict(pred_test5) - y_test5)**2))
#         print(np.mean((cat_model5.predict(pred_test5) - y_test5)**2))
#         print(np.mean((linear_model5.predict(pred_test5) - y_test5)**2))
#         print(ridge_model5.score(pred_test5, y_test5))
#         print(cat_model5.score(pred_test5, y_test5))
#         print(linear_model5.score(pred_test5, y_test5))
#         print(np.mean((ridge_model6.predict(pred_test6) - y_test6)**2))
#         print(np.mean((cat_model6.predict(pred_test6) - y_test6)**2))
#         print(np.mean((linear_model6.predict(pred_test6) - y_test6)**2))
#         print(ridge_model6.score(pred_test6, y_test6))
#         print(cat_model6.score(pred_test6, y_test6))
#         print(linear_model6.score(pred_test6, y_test6))

		mses = [np.mean([np.mean((gaussian_model.predict(pred_test) - y_test)**2), np.mean((gaussian_model1.predict(pred_test1) - y_test1)**2), np.mean((gaussian_model2.predict(pred_test2) - y_test2)**2), np.mean((gaussian_model3.predict(pred_test3) - y_test3)**2), np.mean((gaussian_model4.predict(pred_test4) - y_test4)**2)]),
			np.mean([np.mean((decisiontree_model.predict(pred_test) - y_test)**2), np.mean((decisiontree_model1.predict(pred_test1) - y_test1)**2), np.mean((decisiontree_model2.predict(pred_test2) - y_test2)**2), np.mean((decisiontree_model3.predict(pred_test3) - y_test3)**2), np.mean((decisiontree_model4.predict(pred_test4) - y_test4)**2)]),
			np.mean([np.mean((sgd_model.predict(pred_test) - y_test)**2), np.mean((sgd_model1.predict(pred_test1) - y_test1)**2), np.mean((sgd_model2.predict(pred_test2) - y_test2)**2), np.mean((sgd_model3.predict(pred_test3) - y_test3)**2), np.mean((sgd_model4.predict(pred_test4) - y_test4)**2)]),
			np.mean([np.mean((ridge_model.predict(pred_test) - y_test)**2), np.mean((ridge_model1.predict(pred_test1) - y_test1)**2), np.mean((ridge_model2.predict(pred_test2) - y_test2)**2), np.mean((ridge_model3.predict(pred_test3) - y_test3)**2), np.mean((ridge_model4.predict(pred_test4) - y_test4)**2)]),
			np.mean([np.mean((ard_model.predict(pred_test) - y_test)**2), np.mean((ard_model1.predict(pred_test1) - y_test1)**2), np.mean((ard_model2.predict(pred_test2) - y_test2)**2), np.mean((ard_model3.predict(pred_test3) - y_test3)**2), np.mean((ard_model4.predict(pred_test4) - y_test4)**2)]),
			np.mean([np.mean((bayesian_model.predict(pred_test) - y_test)**2), np.mean((bayesian_model1.predict(pred_test1) - y_test1)**2), np.mean((bayesian_model2.predict(pred_test2) - y_test2)**2), np.mean((bayesian_model3.predict(pred_test3) - y_test3)**2), np.mean((bayesian_model4.predict(pred_test4) - y_test4)**2)]),
			np.mean([np.mean((neural_model.predict(pred_test) - y_test)**2), np.mean((neural_model1.predict(pred_test1) - y_test1)**2), np.mean((neural_model2.predict(pred_test2) - y_test2)**2), np.mean((neural_model3.predict(pred_test3) - y_test3)**2), np.mean((neural_model4.predict(pred_test4) - y_test4)**2)]),
			np.mean([np.mean((cat_model.predict(pred_test) - y_test)**2), np.mean((cat_model1.predict(pred_test1) - y_test1)**2), np.mean((cat_model2.predict(pred_test2) - y_test2)**2), np.mean((cat_model3.predict(pred_test3) - y_test3)**2), np.mean((neural_model4.predict(pred_test4) - y_test4)**2)]),
			np.mean([np.mean((linear_model.predict(pred_test) - y_test)**2), np.mean((linear_model1.predict(pred_test1) - y_test1)**2), np.mean((linear_model2.predict(pred_test2) - y_test2)**2), np.mean((linear_model3.predict(pred_test3) - y_test3)**2), np.mean((linear_model4.predict(pred_test4) - y_test4)**2)])]
		r2s = [np.mean([gaussian_model.score(pred_test, y_test), gaussian_model1.score(pred_test1, y_test1), gaussian_model2.score(pred_test2, y_test2), gaussian_model3.score(pred_test3, y_test3), gaussian_model4.score(pred_test4, y_test4)]),
			np.mean([decisiontree_model.score(pred_test, y_test), decisiontree_model1.score(pred_test1, y_test1), decisiontree_model2.score(pred_test2, y_test2), decisiontree_model3.score(pred_test3, y_test3), decisiontree_model4.score(pred_test4, y_test4)]),
			np.mean([sgd_model.score(pred_test, y_test), sgd_model1.score(pred_test1, y_test1), sgd_model2.score(pred_test2, y_test2), sgd_model3.score(pred_test3, y_test3), sgd_model4.score(pred_test4, y_test4)]),
			np.mean([ridge_model.score(pred_test, y_test), ridge_model1.score(pred_test1, y_test1), ridge_model2.score(pred_test2, y_test2), ridge_model3.score(pred_test3, y_test3), ridge_model4.score(pred_test4, y_test4)]),
			np.mean([ard_model.score(pred_test, y_test), ard_model1.score(pred_test1, y_test1), ard_model2.score(pred_test2, y_test2), ard_model3.score(pred_test3, y_test3), ard_model4.score(pred_test4, y_test4)]),
			np.mean([bayesian_model.score(pred_test, y_test), bayesian_model1.score(pred_test1, y_test1), bayesian_model2.score(pred_test2, y_test2), bayesian_model3.score(pred_test3, y_test3), bayesian_model4.score(pred_test4, y_test4)]),
			np.mean([neural_model.score(pred_test, y_test), neural_model1.score(pred_test1, y_test1), neural_model2.score(pred_test2, y_test2), neural_model3.score(pred_test3, y_test3), neural_model4.score(pred_test4, y_test4)]),
			np.mean([cat_model.score(pred_test, y_test), cat_model1.score(pred_test1, y_test1), cat_model2.score(pred_test2, y_test2), cat_model3.score(pred_test3, y_test3), cat_model4.score(pred_test4, y_test4)]),
			np.mean([linear_model.score(pred_test, y_test), linear_model1.score(pred_test1, y_test1), linear_model2.score(pred_test2, y_test2), linear_model3.score(pred_test3, y_test3), linear_model4.score(pred_test4, y_test4)])]
		
		results_df = results_df.append(pd.Series(mses, index = results_df.columns), ignore_index = True)
		results_df = results_df.append(pd.Series(r2s, index = results_df.columns), ignore_index = True)
		
		print(results_df)
		
		output_filename = "./OutputCSVs/model_testing_" + colname + "_" + str(sample_size) + "_" + datetime.now().strftime("%m_%d_%Y") + ".csv"
		results_df.to_csv(output_filename)
	return results_df

# Here we run cross-validation by testing out combinations of hyperparameters.

def test_model_hyperparameters():
	for i in [6, 7, 8, 9, 10]:
		regression_by_sample(datetime(2018, 11, 2), datetime(2021, 1, 30), i, .85, True)