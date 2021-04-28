from hardcoded import *
from modeling_helpers import *
from processing import *
from modeling import *

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

def make_predictions(start_date, end_date, output = True, per_min = False):
    """Makes predictions for each date between start_date and end_date. Gets box scores for range of dates,
    calls statline_output to make predicted statlines for each game.
    
    Params:
    start_date: Datetime object representing date to begin collecting data from.
    end_date: Datetime object representing last date to collect data for.
    output: Boolean, whether to write predictions to file.
    """
    full_df = box_scores_for_range_of_days(start_date, end_date)
    predicted_statlines = statline_output(full_df, input_statistics, per_min)
    if output:
        output_filename = './AllCSVs/' + str(start_date.month) + '_' + str(start_date.day) + '_' + str(end_date.month) + '_' + str(end_date.day) + '_' + str(end_date.year) + '_predicted_box_scores.csv'
        predicted_statlines.to_csv(output_filename)
    return predicted_statlines

def predict_next_day(start_date, game_date):
    """Makes predictions for game_date. Calls make_predictions to produce predictions for game_date and
    keeps players who are playing on game_date. Writes predictions to file.


    Params:
    start_date: Datetime object representing date to begin collecting data from.
    game_date: Datetime object representing date to produce outputs for.
    """
    season_schedule = client.season_schedule(season_end_year=game_date.year)
    schedule_on_date = [game for game in season_schedule if
                        (game['start_time'] - timedelta(hours=4)).day == game_date.day and game[
                            'start_time'].month == game_date.month]
    team_objs_on_date = [game["home_team"] for game in schedule_on_date] + [game["away_team"] for game in
                                                                            schedule_on_date]
    teams_on_date = [t.name.replace("_", " ") for t in team_objs_on_date]
    predictions_for_date = make_predictions(start_date, game_date, output=False)
    players_on_date = predictions_for_date[predictions_for_date.team.isin(teams_on_date)]
    mstr, dstr = str(game_date.month), str(game_date.day)
    if game_date.month < 10:
        mstr = "0" + mstr
    if game_date.day < 10:
        dstr = "0" + dstr
    tstr = mstr + "-" + dstr
    statlines_on_date = players_on_date[players_on_date.date.str.contains(tstr)]
    statlines_on_date.index = range(statlines_on_date.shape[0])
    output_filename = './AllCSVs/predictions_for_' + mstr + "_" + dstr + '_from_' + str(start_date.month) + '_' + str(
        start_date.day) + '_' + str(start_date.year) + '.csv'
    statlines_on_date.to_csv(output_filename)
    return statlines_on_date


def append_actual_results(start_date, game_date):
    """Created predicted statlines for game_date and, if actual statlines exist, attaches
    actual statlines to predicted statlines for use in testing. Writes output to file.

    Params:
    start_date: Datetime object representing date to begin collecting data from.
    game_date: Datetime object representing date to produce outputs for.
    """
    predicted_results = predict_next_day(start_date, game_date)
    actual_results = box_scores_for_range_of_days(game_date, game_date)
    actual_results["made_two_point_field_goals"] = actual_results["made_field_goals"] - actual_results[
        "made_three_point_field_goals"]
    actual_results["rebounds"] = actual_results["offensive_rebounds"] + actual_results["defensive_rebounds"]
    pred_outputs = [float(get_points(predicted_results[predicted_results["name"] == player_name])[0]) for player_name in
                    predicted_results["name"]]
    actual_outputs = [float(get_points(actual_results[actual_results["name"] == player_name])[0]) for player_name in
                      predicted_results["name"]]
    predicted_results["predicted_points"] = pred_outputs
    predicted_results["actual_points"] = actual_outputs
    output_filename = './AllCSVs/' + str(game_date.month) + '_' + str(game_date.day) + '_' + str(
        game_date.year) + '_box_scores_predicted_with_actual.csv'
    predicted_results.to_csv(output_filename)
    return predicted_results


def alternate_contests(start_date, game_date, search_datetime, inclusive):
    """Makes predictions for game_date, only for game times occurring after search_datetime.
    If inclusive is true, makes predictions for game times on or after search_datetime.

    Params:
    start_date: Datetime object representing date to begin collecting data from.
    game_date: Datetime object representing date to produce outputs for.
    search_datetime: Datetime object representing alternate game start time to filter for.
    inclusive: Boolean, determines whether games starting exactly at search_datetime are included.
    """
    season_schedule = client.season_schedule(season_end_year=game_date.year)
    schedule_on_date = [game for game in season_schedule if
                        (game['start_time'] - timedelta(hours=4)).day == game_date.day and (
                                    game['start_time'] - timedelta(hours=4)).month == game_date.month]
    times_on_date = np.unique([game['start_time'] - timedelta(hours=4) for game in schedule_on_date])
    print(times_on_date)
    if not inclusive:
        game_teams = [game["home_team"] for game in schedule_on_date if
                      (game['start_time'] - timedelta(hours=4)) == search_datetime] + [game["away_team"] for game in
                                                                                       schedule_on_date if (game[
                                                                                                                'start_time'] - timedelta(
                hours=4)) == search_datetime]
    else:
        times_after_search = [t for t in times_on_date if
                              60 * t.hour + t.minute >= 60 * search_datetime.hour + search_datetime.minute]
        game_teams = [game["home_team"] for game in schedule_on_date if
                      (game['start_time'] - timedelta(hours=4)) in times_after_search] + [game["away_team"] for game in
                                                                                          schedule_on_date if (game[
                                                                                                                   'start_time'] - timedelta(
                hours=4)) in times_after_search]
    game_team_strings = [str(team)[5:].replace("_", " ") for team in game_teams]
    predictions_for_date = predict_next_day(start_date, game_date)
    players_on_date = predictions_for_date[predictions_for_date.team.isin(game_team_strings)]
    output_filename = './AllCSVs/' + str(game_date.month) + '_' + str(game_date.day) + '_' + str(
        game_date.year) + '_alternate_' + str(search_datetime.hour) + "_" + str(search_datetime.minute) + '.csv'
    players_on_date.to_csv(output_filename)
    return players_on_date

def generate_optimal_lineup(players, positions, values, costs, budget):
    """Generates the optimal lineup given a list of players, their positions, their modeled Fanduel point values,
    their Fanduel salaries and the budget for this lineup.
    
    Params:
    players: Array of strings representing player names.
    positions: Array of strings representing player positions.
    values: Array of float modeled Fanduel values for each player.
    costs: Array of integer salaries for each player.
    budget: Integer, maximum sum of costs in lineup.
    """
    num_variables = len(players)
    
    lp = LpProblem("My LP Problem", LpMaximize)
    
    d = {}
    for i in range(0, num_variables):
        d[players[i]] = LpVariable(players[i], cat="Binary")
    
    objective = sum(np.array(values) * np.array(list(d.values())))
    lp += objective
    
    pg_constraint = 0
    sg_constraint = 0
    sf_constraint = 0
    pf_constraint = 0
    c_constraint = 0
    for i in range(0, len(positions)):
        if positions[i] == "PG":
            pg_constraint += d[players[i]]
        elif positions[i] == "SG":
            sg_constraint += d[players[i]]
        elif positions[i] == "SF":
            sf_constraint += d[players[i]]
        elif positions[i] == "PF":
            pf_constraint += d[players[i]]
        else:
            c_constraint += d[players[i]]
    lp += pg_constraint == 2
    lp += sg_constraint == 2
    lp += sf_constraint == 2
    lp += pf_constraint == 2
    lp += c_constraint == 1
    
    cost = sum(np.array(costs) * np.array(list(d.values())))
    lp += cost <= budget
    
    lp.solve()
    
    lineup = [variable.name for variable in lp.variables() if variable.varValue == 1]
    return lineup

def generate_optimal_lineup_draftkings(players, positions, values, costs, budget):
    """See above documentation for generate_optimal_lineup. Draftkings equivalent.
    """
    num_variables = len(players)
    
    lp = LpProblem("My LP Problem", LpMaximize)
    
    d = {}
    for i in range(0, num_variables):
        d[players[i]] = LpVariable(players[i], cat="Binary")
    
    objective = sum(np.array(values) * np.array(list(d.values())))
    lp += objective
    
    pg_constraint = 0
    sg_constraint = 0
    sf_constraint = 0
    pf_constraint = 0
    c_constraint = 0
    g_constraint = 0
    f_constraint = 0
    player_constraint = 0
    for i in range(0, len(positions)):
        if "PG" in positions[i] or "SG" in positions[i]:
            if "PG" in positions[i]:
                pg_constraint += d[players[i]]
            if "SG" in positions[i]:
                sg_constraint += d[players[i]]
            g_constraint += d[players[i]]
        if "PF" in positions[i] or "SF" in positions[i]:
            if "PF" in positions[i]:
                pf_constraint += d[players[i]]
            if "SF" in positions[i]:
                sf_constraint += d[players[i]]
            f_constraint += d[players[i]]
        if "C" in positions[i]:
            c_constraint += d[players[i]]
        player_constraint += d[players[i]]
    lp += pg_constraint <= 3
    lp += sg_constraint <= 3
    lp += sf_constraint <= 3
    lp += pf_constraint <= 3
    lp += pg_constraint >= 1
    lp += sg_constraint >= 1
    lp += sf_constraint >= 1
    lp += pf_constraint >= 1
    lp += c_constraint >= 1
    lp += g_constraint <= 4
    lp += f_constraint <= 4
    lp += g_constraint >= 3
    lp += f_constraint >= 3
    lp += c_constraint <= 2
    lp += player_constraint == 8
    
    cost = sum(np.array(costs) * np.array(list(d.values())))
    lp += cost <= budget
    
    lp.solve()
    
    lineup = [variable.name for variable in lp.variables() if variable.varValue == 1]
    return lineup

def build_n_lineups_fanduel(players, positions, values, costs, budget, n, sd_param = .35):
    lineups = []
    players_in = {}
    for i in range(n):
        randomness = np.random.normal(loc = 1, scale = sd_param, size = len(values))
        values = values * randomness
        lineup = [n.replace("_", " ") for n in generate_optimal_lineup(players, positions, values, costs, budget)]
        for player in lineup:
            if player not in players_in.keys():
                players_in[player] = 1
            else:
                players_in[player] += 1
        lineups.append(lineup)
    return lineups, players_in
    
def build_n_lineups_draftkings(players, positions, values, costs, budget, n, sd_param = .35):
    lineups = []
    players_in = {}
    for i in range(n):
        randomness = np.random.normal(loc = 1, scale = sd_param, size = len(values))
        values = values * randomness
        lineup = [n.replace("_", " ") for n in generate_optimal_lineup_draftkings(players, positions, values, costs, budget)]
        for player in lineup:
            if player not in players_in.keys():
                players_in[player] = 1
            else:
                players_in[player] += 1
        lineups.append(lineup)
    return lineups, players_in

def predict_unplayed_games(start_date, game_date, retrain = False, pretrain_inputs = True):
    """Makes predictions for unplayed games on game_date, using data beginning at start_date to make its
    predictions. Finds the date's schedule, predicts statlines for each player on the competing teams, and
    writes the output predictions to file.
    
    Params:
    start_date: Datetime object representing date to begin collecting data from.
    game_date: Datetime object representing date to produce outputs for.
    """
    season_schedule = client.season_schedule(season_end_year = game_date.year)
    schedule_on_date = [game for game in season_schedule if (game['start_time'] - timedelta(hours = 4)).day == game_date.day and (game['start_time'] - timedelta(hours = 4)).month == game_date.month]
    available_box_scores = box_scores_for_range_of_days(start_date, game_date)
    game_teams = [game["home_team"] for game in schedule_on_date] + [game["away_team"] for game in schedule_on_date]
    matchups = [[str(game["home_team"])[5:].replace("_", " "), str(game["away_team"])[5:].replace("_", " ")] for game in schedule_on_date]
    game_team_strings = [str(team)[5:].replace("_", " ") for team in game_teams]
    relevant_lines = available_box_scores[available_box_scores.team.isin(game_team_strings)]
    relevant_players = relevant_lines.drop_duplicates(subset = ["name"], keep = 'last')
    mstr, dstr = str(game_date.month), str(game_date.day)
    if game_date.month < 10:
        mstr = "0" + mstr
    if game_date.day < 10:
        dstr = "0" + dstr
    tstr = str(game_date.year) + "-" + mstr + "-" + dstr
    relevant_players.Date = [tstr for _ in range(len(relevant_players))]
    relevant_players.opponent = [matchup_lookup(matchups, t) for t in relevant_players.team]
    augmented_box_scores = available_box_scores.append(relevant_players)
    augmented_box_scores.index = range(augmented_box_scores.shape[0])
    predicted_statlines = statline_output(augmented_box_scores, input_statistics, retrain = retrain, pretrain_inputs = pretrain_inputs)
    statlines_on_date = predicted_statlines[predicted_statlines.date.str.contains(tstr)]
    statlines_on_date['projected_points'] = [round(float(get_points(statlines_on_date[statlines_on_date["name"] == player_name])[0]), 2) for player_name in statlines_on_date.name]
    statlines_on_date['projected_points_draftkings'] = [round(float(get_draftkings_points(statlines_on_date[statlines_on_date["name"] == player_name])[0]), 2) for player_name in statlines_on_date.name]
    statlines_on_date['projected_value'] = [round(float(get_points(statlines_on_date[statlines_on_date["name"] == player_name])[1])) for player_name in statlines_on_date.name]
    statlines_on_date['projected_value_draftkings'] = [round(float(get_draftkings_points(statlines_on_date[statlines_on_date["name"] == player_name])[1])) for player_name in statlines_on_date.name]
    output_filename = './AllCSVs/predictions_for_' + mstr + "_" + dstr + "_" + str(game_date.year) + "_unplayed.csv"
    statlines_on_date.to_csv(output_filename)
    return statlines_on_date

def per_minute_projections(projections):
    statline_categories = ['2PT FG', '3PT FG', 'FTM', 'Rebounds', 'Assists', 'Blocks', 'Steals', 'Turnovers']
    statline = projections.loc[:, statline_categories]
    per_minute_statline = statline.div(projections['Minutes'], axis=0)
    projections.loc[:,statline_categories] = per_minute_statline
    projections = projections.drop(["Minutes"], axis=1)
    projections["FanDuel Points Per Minute"] = 2 * projections["2PT FG"] + 3 * projections["3PT FG"] + projections["FTM"] + 1.2 * projections["Rebounds"] + 1.5 * projections["Assists"] + 3 * projections["Blocks"] + 3 * projections["Steals"] - projections["Turnovers"]
    projections["DraftKings Points Per Minute"] = 2 * projections["2PT FG"] + 3.5 * projections["3PT FG"] + projections["FTM"] + 1.25 * projections["Rebounds"] + 1.5 * projections["Assists"] + 2 * projections["Blocks"] + 2 * projections["Steals"] - .5 * projections["Turnovers"]
    projections = projections[['Name', 'Position', 'Game', 'Team',
        'FanDuel Points Per Minute', 'DraftKings Points Per Minute', 
        '2PT FG', '3PT FG', 'FTM', 'Rebounds', 'Assists', 'Blocks', 
        'Steals', 'Turnovers', 'Opponent Defensive Rank vs Position', 
        'Injury Indicator', 'Injury Details']]
    return projections

def optimal_lineup_fanduel_games(game_date, predictions, fanduel_csv, players_out = []):
    """Given a game_date, a csv of predicted statlines, and Fanduel's csv containing data for this contest including
    player salaries and injury information, produces the optimal lineup for Fanduel. Injured players should be
    manually entered into players_out so as to exclude them from the optimal lineup.
    
    Params:
    game_date: Datetime object representing date to produce outputs for.
    predictions: Predicted statlines for the game_date. Should have a csv written to file that is generated
        by calling predict_unplayed_games earlier.
    fanduel_csv: CSV of data for the Fanduel contest.
    players_out: List of players who will not be available. Usually manually updated, if this function produces
        an optimal lineup with an injured player, add the player's name to players_out and run the function again.
    """
    fanduel_data = pd.read_csv(fanduel_csv).fillna({"FPPG": 0})
    print("Done predicting!")
    for i in predictions.index:
        if predictions.loc[i, "name"] in difficult_names_map_fanduel.keys():
            n = predictions.loc[i, "name"]
            predictions.loc[i, "name"] = difficult_names_map_fanduel[n]
    fanduel_data_with_predictions = fanduel_data.merge(predictions, left_on = "Nickname", right_on = "name").fillna(" ")
    fanduel_data_with_predictions["FPPG"] = fanduel_data_with_predictions["FPPG"].round(2)
    fanduel_data_with_predictions["Value above Market Value"] = fanduel_data_with_predictions["projected_value"] - fanduel_data_with_predictions["Salary"]
    fanduel_data_with_predictions["Value above Market Value"] = np.where(fanduel_data_with_predictions["Value above Market Value"] < 0, '-$' + fanduel_data_with_predictions["Value above Market Value"].astype(str).str[1:], '$' + fanduel_data_with_predictions["Value above Market Value"].astype(str))
    fanduel_data_with_predictions_injuries = fanduel_data_with_predictions[fanduel_data_with_predictions["Injury Indicator"] != "O"]
    players_not_out = [(name not in players_out) for name in fanduel_data_with_predictions_injuries["Nickname"]]
    fanduel_data_with_predictions_injuries = fanduel_data_with_predictions_injuries[players_not_out]
    fanduel_data_with_predictions_injuries.index = range(fanduel_data_with_predictions_injuries.shape[0])
    #print(fanduel_data_with_predictions_injuries)
    values = fanduel_data_with_predictions_injuries["projected_points"]
    players = fanduel_data_with_predictions_injuries["Nickname"]
    positions = fanduel_data_with_predictions_injuries["Position"]
    costs = fanduel_data_with_predictions_injuries["Salary"]
    data_to_feed = pd.DataFrame(data = {'players': players, 'positions': positions, 'values': values, 'costs': costs})
    optimal_lineup = [n.replace("_", " ") for n in generate_optimal_lineup(players, positions, values, costs, 60000)]
    print(optimal_lineup)
    for i in range(len(optimal_lineup)):
        if optimal_lineup[i] in punctuation_names.keys():
            optimal_lineup[i] = punctuation_names[optimal_lineup[i]]
    fanduel_data_with_predictions_injuries["projected_value"] = ['$' + str(s) for s in fanduel_data_with_predictions_injuries["projected_value"]]
    fanduel_data_with_predictions_injuries["Salary"] = ['$' + str(s) for s in fanduel_data_with_predictions_injuries["Salary"]]
    projs_in_optimal = [(fanduel_data_with_predictions_injuries.loc[i, "Nickname"] in optimal_lineup) for i in fanduel_data_with_predictions_injuries.index]
    lineup_return = fanduel_data_with_predictions_injuries[projs_in_optimal].sort_values(by=["Position"], ascending = False)
    lineup_return = lineup_return.loc[:, ["Nickname", "Position", "Game", "Team", 'projected_points', 'projected_value', "Salary", "Value above Market Value", 'projected_points_draftkings', 'projected_value_draftkings', "FPPG", '10_game_average', '3_game_average', "Injury Indicator", "Injury Details", "minutes", "made_two_point_field_goals", 'made_three_point_field_goals',
       'made_free_throws', 'rebounds', 'assists', 'blocks', 'steals', 'turnovers','hot', 'cold']]
    lineup_return.columns = ["Name", "Position", "Game", "Team", 'Projected Fanduel Points', 'Projected Fanduel Value', "Fanduel Salary",  "Value above Fanduel Value", 'Projected Draftkings Points', 'Projected Draftkings Value', "FPPG (Fanduel)", '10 Game Average (Fanduel)', '3 Game Average (Fanduel)', "Injury Indicator", "Injury Details", "Minutes", "2PT FG", '3PT FG',
       'FTM', 'Rebounds', 'Assists', 'Blocks', 'Steals', 'Turnovers','Hot', 'Cold']
    csv_return = fanduel_data_with_predictions.loc[:, ["Nickname", "Position", "Game", "Team", 'projected_points', 'projected_value', "Salary", "Value above Market Value", 'projected_points_draftkings', 'projected_value_draftkings', "FPPG", '10_game_average', '3_game_average', "Injury Indicator", "Injury Details", "minutes", "made_two_point_field_goals", 'made_three_point_field_goals',
       'made_free_throws', 'rebounds', 'assists', 'blocks', 'steals', 'turnovers', "Opponent Defensive Rank vs Position", 'hot', 'cold']]
    csv_return.columns = ["Name", "Position", "Game", "Team", 'Projected Fanduel Points', 'Projected Fanduel Value', "Fanduel Salary", "Value above Fanduel Value", 'Projected Draftkings Points', 'Projected Draftkings Value', "FPPG (Fanduel)", '10 Game Average (Fanduel)', '3 Game Average (Fanduel)', "Injury Indicator", "Injury Details", "Minutes", "2PT FG", '3PT FG',
       'FTM', 'Rebounds', 'Assists', 'Blocks', 'Steals', 'Turnovers', "Opponent Defensive Rank vs Position", 'Hot', 'Cold']
    csv_return_dfs = csv_return.loc[:, ["Name", "Position", "Game", "Team", 'Projected Fanduel Points', 'Projected Fanduel Value', "Fanduel Salary", "Value above Fanduel Value", 'Projected Draftkings Points', 'Projected Draftkings Value', "FPPG (Fanduel)", '10 Game Average (Fanduel)', '3 Game Average (Fanduel)', "Opponent Defensive Rank vs Position",  "Injury Indicator", "Injury Details", 'Hot', 'Cold']]
    csv_return_projections = csv_return.loc[:, ["Name", "Position", "Game", "Team", "Minutes", "2PT FG", '3PT FG',
       'FTM', 'Rebounds', 'Assists', 'Blocks', 'Steals', 'Turnovers', "Opponent Defensive Rank vs Position", "Injury Indicator", "Injury Details"]]
    lineup_return = lineup_return.loc[:, ["Name", "Position", "Game", "Team", 'Projected Fanduel Points', 'Projected Fanduel Value', "Fanduel Salary", "Value above Fanduel Value", "FPPG (Fanduel)", "Injury Indicator", "Injury Details", 'Hot', 'Cold']]
    
    csv_return_dfs.to_csv("./AllCSVs/dfs_projections_to_display_" + str(game_date.month) + "_" + str(game_date.day) + "_" + str(game_date.year) + ".csv")
    csv_return_projections.to_csv("./AllCSVs/statline_projections_to_display_" + str(game_date.month) + "_" + str(game_date.day) + "_" + str(game_date.year) + ".csv")
    lineup_return.to_csv("./AllCSVs/optimal_fanduel_lineup_" + str(game_date.month) + "_" + str(game_date.day) + "_" + str(game_date.year) + ".csv")
    return csv_return_dfs, csv_return_projections, lineup_return

def optimal_lineup_draftkings_games(game_date, predictions, draftkings_csv, players_out = []):
    """See above documentation for optimal_lineup_fanduel_games. Draftkings equivalent."""
    draftkings_data = pd.read_csv(draftkings_csv)
    print("Done predicting!")
    for i in predictions.index:
        if predictions.loc[i, "name"] in difficult_names_map_draftkings.keys():
            n = predictions.loc[i, "name"]
            predictions.loc[i, "name"] = difficult_names_map_draftkings[n]
    draftkings_data_with_predictions = draftkings_data.merge(predictions, left_on = "Name", right_on = "name")
    draftkings_data_with_predictions["AvgPointsPerGame"] = draftkings_data_with_predictions["AvgPointsPerGame"].round(2)
    draftkings_data_with_predictions["Game Info"] = [g[:7] for g in draftkings_data_with_predictions["Game Info"]]
    draftkings_data_with_predictions["Value above Market Value"] = draftkings_data_with_predictions["projected_value_draftkings"] - draftkings_data_with_predictions["Salary"]
    draftkings_data_with_predictions["Value above Market Value"] = np.where(draftkings_data_with_predictions["Value above Market Value"] < 0, '-$' + draftkings_data_with_predictions["Value above Market Value"].astype(str).str[1:], '$' + draftkings_data_with_predictions["Value above Market Value"].astype(str))
    draftkings_data_with_predictions_injuries = draftkings_data_with_predictions#[draftkings_data_with_predictions["Injury Indicator"] != "O"]
    players_not_out = [(name not in players_out) for name in draftkings_data_with_predictions_injuries["Name"]]
    draftkings_data_with_predictions_injuries = draftkings_data_with_predictions_injuries[players_not_out]
    draftkings_data_with_predictions_injuries.index = range(draftkings_data_with_predictions_injuries.shape[0])
    #print(fanduel_data_with_predictions_injuries)
    draftkings_data_with_predictions_injuries.index = range(len(draftkings_data_with_predictions_injuries))
    values = draftkings_data_with_predictions_injuries["projected_points_draftkings"]
    players = draftkings_data_with_predictions_injuries["Name"]
    positions_as_lists = [list(st.split("/")) for st in draftkings_data_with_predictions_injuries["Position"]]
    positions = draftkings_data_with_predictions_injuries["Position"]
    costs = draftkings_data_with_predictions_injuries["Salary"]
    data_to_feed = pd.DataFrame(data = {'players': players, 'positions': positions_as_lists, 'values': values, 'costs': costs})
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(data_to_feed)
    optimal_lineup = [n.replace("_", " ") for n in generate_optimal_lineup_draftkings(players, positions_as_lists, values, costs, 50000)]
    print(optimal_lineup)
    for i in range(len(optimal_lineup)):
        if optimal_lineup[i] in punctuation_names.keys():
            optimal_lineup[i] = punctuation_names[optimal_lineup[i]]
    draftkings_data_with_predictions_injuries["projected_value_draftkings"] = ['$' + str(s) for s in draftkings_data_with_predictions_injuries["projected_value_draftkings"]]
    draftkings_data_with_predictions_injuries["Salary"] = ['$' + str(s) for s in draftkings_data_with_predictions_injuries["Salary"]]
    projs_in_optimal = [(draftkings_data_with_predictions_injuries.loc[i, "Name"] in optimal_lineup) for i in draftkings_data_with_predictions_injuries.index]
    lineup_return = draftkings_data_with_predictions_injuries[projs_in_optimal].sort_values(by=["Position"], ascending = False)
    lineup_return = lineup_return.loc[:, ["Name", "Position", "Game Info", "TeamAbbrev", 'projected_points', 'projected_value', 'projected_points_draftkings', 'projected_value_draftkings', "Salary", "Value above Market Value", "AvgPointsPerGame", '10_game_average', '3_game_average', "minutes", "made_two_point_field_goals", 'made_three_point_field_goals',
       'made_free_throws', 'rebounds', 'assists', 'blocks', 'steals', 'turnovers','hot', 'cold']].fillna(" ")
    lineup_return.columns = ["Name", "Position", "Game", "Team", 'Projected Fanduel Points', 'Projected Fanduel Value', 'Projected Draftkings Points', 'Projected Draftkings Value', "Draftkings Salary", "Value above Draftkings Value", "FPPG", '10 Game Average (Fanduel)', '3 Game Average (Fanduel)', "Minutes", "2PT FG", '3PT FG',
       'FTM', 'Rebounds', 'Assists', 'Blocks', 'Steals', 'Turnovers','Hot', 'Cold']
    csv_return = draftkings_data_with_predictions.loc[:, ["Name", "Position", "Salary", "Game Info", "TeamAbbrev", 'projected_points', 'projected_value', "Value above Market Value", 'projected_points_draftkings', 'projected_value_draftkings', "AvgPointsPerGame", '10_game_average', '3_game_average', "minutes", "made_two_point_field_goals", 'made_three_point_field_goals',
       'made_free_throws', 'rebounds', 'assists', 'blocks', 'steals', 'turnovers', "Opponent Defensive Rank vs Position", 'hot', 'cold']].fillna(" ")
    csv_return.columns = ["Name", "Position", "Salary", "Game", "Team", 'Projected Fanduel Points', 'Projected Fanduel Value', "Value above Fanduel Value", 'Projected Draftkings Points', 'Projected Draftkings Value', "FPPG", '10 Game Average (Fanduel)', '3 Game Average (Fanduel)', "Minutes", "2PT FG", '3PT FG',
       'FTM', 'Rebounds', 'Assists', 'Blocks', 'Steals', 'Turnovers', "Opponent Defensive Rank vs Position", 'Hot', 'Cold']
    csv_return_dfs = csv_return.loc[:, ["Name", "Position", "Salary", "Game", "Team", 'Projected Fanduel Points', 'Projected Fanduel Value', "Value above Fanduel Value", 'Projected Draftkings Points', 'Projected Draftkings Value', "FPPG", '10 Game Average (Fanduel)', '3 Game Average (Fanduel)', "Opponent Defensive Rank vs Position", 'Hot', 'Cold']]
    csv_return_projections = csv_return.loc[:, ["Name", "Position", "Game", "Team", "Minutes", "2PT FG", '3PT FG',
       'FTM', 'Rebounds', 'Assists', 'Blocks', 'Steals', 'Turnovers', "Opponent Defensive Rank vs Position"]]
    lineup_return = lineup_return.loc[:, ["Name", "Position", "Game", "Team", 'Projected Draftkings Points', 'Projected Draftkings Value', "Draftkings Salary", "Value above Draftkings Value", "FPPG", 'Hot', 'Cold']]
    
    fanduel_statlines = pd.read_csv("./AllCSVs/statline_projections_to_display_" + str(game_date.month) + "_" + str(game_date.day) + "_" + str(game_date.year) + ".csv")
    injury_indicators = []
    injury_details = []
    for i in lineup_return.index:
        print(lineup_return.loc[i, "Name"])
        for n in fanduel_statlines.index:
            if fanduel_statlines.loc[n, "Name"] in fd_to_dk.keys():
                nam = fd_to_dk[fanduel_statlines.loc[n, "Name"]]
                fanduel_statlines.loc[n, "Name"] = nam
            if fanduel_statlines.loc[n, "Name"] == lineup_return.loc[i, "Name"]:
                print(i)
                injury_indicators.append(fanduel_statlines.loc[n, "Injury Indicator"])
                injury_details.append(fanduel_statlines.loc[n, "Injury Details"])
            elif lineup_return.loc[i, "Name"] in punctuation_names.keys():
                if punctuation_names[lineup_return.loc[i, "Name"]] == fanduel_statline.loc[n, "Name"]:
                    print(i)
                    injury_indicators.append(fanduel_statlines.loc[n, "Injury Indicator"])
                    injury_details.append(fanduel_statlines.loc[n, "Injury Details"])
    lineup_return["Injury Indicator"] = injury_indicators
    lineup_return["Injury Details"] = injury_details
    
    lineup_return.to_csv("./AllCSVs/optimal_draftkings_lineup_" + str(game_date.month) + "_" + str(game_date.day) + "_" + str(game_date.year) + ".csv")
    return csv_return_dfs, csv_return_projections, lineup_return

def gen_n_lineups_draftkings(game_date, predictions, draftkings_csv, players_out = [], n = 100):
    """See above documentation for optimal_lineup_fanduel_games. Draftkings equivalent."""
    draftkings_data = pd.read_csv(draftkings_csv)
    print("Done predicting!")
    for i in predictions.index:
        if predictions.loc[i, "name"] in difficult_names_map_draftkings.keys():
            n = predictions.loc[i, "name"]
            predictions.loc[i, "name"] = difficult_names_map_draftkings[n]
    draftkings_data_with_predictions = draftkings_data.merge(predictions, left_on = "Name", right_on = "name")
    draftkings_data_with_predictions["AvgPointsPerGame"] = draftkings_data_with_predictions["AvgPointsPerGame"].round(2)
    draftkings_data_with_predictions["Game Info"] = [g[:7] for g in draftkings_data_with_predictions["Game Info"]]
    draftkings_data_with_predictions["Value above Market Value"] = draftkings_data_with_predictions["projected_value_draftkings"] - draftkings_data_with_predictions["Salary"]
    draftkings_data_with_predictions["Value above Market Value"] = np.where(draftkings_data_with_predictions["Value above Market Value"] < 0, '-$' + draftkings_data_with_predictions["Value above Market Value"].astype(str).str[1:], '$' + draftkings_data_with_predictions["Value above Market Value"].astype(str))
    draftkings_data_with_predictions_injuries = draftkings_data_with_predictions#[draftkings_data_with_predictions["Injury Indicator"] != "O"]
    players_not_out = [(name not in players_out) for name in draftkings_data_with_predictions_injuries["Name"]]
    draftkings_data_with_predictions_injuries = draftkings_data_with_predictions_injuries[players_not_out]
    draftkings_data_with_predictions_injuries.index = range(draftkings_data_with_predictions_injuries.shape[0])
    #print(fanduel_data_with_predictions_injuries)
    draftkings_data_with_predictions_injuries.index = range(len(draftkings_data_with_predictions_injuries))
    values = draftkings_data_with_predictions_injuries["projected_points_draftkings"]
    players = draftkings_data_with_predictions_injuries["Name"]
    positions_as_lists = [list(st.split("/")) for st in draftkings_data_with_predictions_injuries["Position"]]
    positions = draftkings_data_with_predictions_injuries["Position"]
    costs = draftkings_data_with_predictions_injuries["Salary"]
    data_to_feed = pd.DataFrame(data = {'players': players, 'positions': positions_as_lists, 'values': values, 'costs': costs})
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(data_to_feed)
    optimal_lineup = [n.replace("_", " ") for n in generate_optimal_lineup_draftkings(players, positions_as_lists, values, costs, 50000)]
    lineups_100, players_in = build_n_lineups_draftkings(players, positions_as_lists, values, costs, budget = 50000, n = n)
    print(optimal_lineup)
    for i in range(len(optimal_lineup)):
        if optimal_lineup[i] in punctuation_names.keys():
            optimal_lineup[i] = punctuation_names[optimal_lineup[i]]
    draftkings_data_with_predictions_injuries["projected_value_draftkings"] = ['$' + str(s) for s in draftkings_data_with_predictions_injuries["projected_value_draftkings"]]
    draftkings_data_with_predictions_injuries["Salary"] = ['$' + str(s) for s in draftkings_data_with_predictions_injuries["Salary"]]
    projs_in_optimal = [(draftkings_data_with_predictions_injuries.loc[i, "Name"] in optimal_lineup) for i in draftkings_data_with_predictions_injuries.index]
    lineup_return = draftkings_data_with_predictions_injuries[projs_in_optimal].sort_values(by=["Position"], ascending = False)
    lineup_return = lineup_return.loc[:, ["Name", "Position", "Game Info", "TeamAbbrev", 'projected_points', 'projected_value', 'projected_points_draftkings', 'projected_value_draftkings', "Salary", "Value above Market Value", "AvgPointsPerGame", '10_game_average', '3_game_average', "minutes", "made_two_point_field_goals", 'made_three_point_field_goals',
       'made_free_throws', 'rebounds', 'assists', 'blocks', 'steals', 'turnovers','hot', 'cold']].fillna(" ")
    lineup_return.columns = ["Name", "Position", "Game", "Team", 'Projected Fanduel Points', 'Projected Fanduel Value', 'Projected Draftkings Points', 'Projected Draftkings Value', "Draftkings Salary", "Value above Draftkings Value", "FPPG", '10 Game Average (Fanduel)', '3 Game Average (Fanduel)', "Minutes", "2PT FG", '3PT FG',
       'FTM', 'Rebounds', 'Assists', 'Blocks', 'Steals', 'Turnovers','Hot', 'Cold']
    csv_return = draftkings_data_with_predictions.loc[:, ["Name", "Position", "Salary", "Game Info", "TeamAbbrev", 'projected_points', 'projected_value', "Value above Market Value", 'projected_points_draftkings', 'projected_value_draftkings', "AvgPointsPerGame", '10_game_average', '3_game_average', "minutes", "made_two_point_field_goals", 'made_three_point_field_goals',
       'made_free_throws', 'rebounds', 'assists', 'blocks', 'steals', 'turnovers', "Opponent Defensive Rank vs Position", 'hot', 'cold']].fillna(" ")
    csv_return.columns = ["Name", "Position", "Salary", "Game", "Team", 'Projected Fanduel Points', 'Projected Fanduel Value', "Value above Fanduel Value", 'Projected Draftkings Points', 'Projected Draftkings Value', "FPPG", '10 Game Average (Fanduel)', '3 Game Average (Fanduel)', "Minutes", "2PT FG", '3PT FG',
       'FTM', 'Rebounds', 'Assists', 'Blocks', 'Steals', 'Turnovers', "Opponent Defensive Rank vs Position", 'Hot', 'Cold']
    csv_return_dfs = csv_return.loc[:, ["Name", "Position", "Salary", "Game", "Team", 'Projected Fanduel Points', 'Projected Fanduel Value', "Value above Fanduel Value", 'Projected Draftkings Points', 'Projected Draftkings Value', "FPPG", '10 Game Average (Fanduel)', '3 Game Average (Fanduel)', "Opponent Defensive Rank vs Position", 'Hot', 'Cold']]
    csv_return_projections = csv_return.loc[:, ["Name", "Position", "Game", "Team", "Minutes", "2PT FG", '3PT FG',
       'FTM', 'Rebounds', 'Assists', 'Blocks', 'Steals', 'Turnovers', "Opponent Defensive Rank vs Position"]]
    lineup_return = lineup_return.loc[:, ["Name", "Position", "Game", "Team", 'Projected Draftkings Points', 'Projected Draftkings Value', "Draftkings Salary", "Value above Draftkings Value", "FPPG", 'Hot', 'Cold']]
    
    fanduel_statlines = pd.read_csv("./AllCSVs/statline_projections_to_display_" + str(game_date.month) + "_" + str(game_date.day) + "_" + str(game_date.year) + ".csv")
    injury_indicators = []
    injury_details = []
    for i in lineup_return.index:
        print(lineup_return.loc[i, "Name"])
        for n in fanduel_statlines.index:
            if fanduel_statlines.loc[n, "Name"] in fd_to_dk.keys():
                nam = fd_to_dk[fanduel_statlines.loc[n, "Name"]]
                fanduel_statlines.loc[n, "Name"] = nam
            if fanduel_statlines.loc[n, "Name"] == lineup_return.loc[i, "Name"]:
                print(i)
                injury_indicators.append(fanduel_statlines.loc[n, "Injury Indicator"])
                injury_details.append(fanduel_statlines.loc[n, "Injury Details"])
            elif lineup_return.loc[i, "Name"] in punctuation_names.keys():
                if punctuation_names[lineup_return.loc[i, "Name"]] == fanduel_statline.loc[n, "Name"]:
                    print(i)
                    injury_indicators.append(fanduel_statlines.loc[n, "Injury Indicator"])
                    injury_details.append(fanduel_statlines.loc[n, "Injury Details"])
    lineup_return["Injury Indicator"] = injury_indicators
    lineup_return["Injury Details"] = injury_details
    
    lineup_return.to_csv("./AllCSVs/optimal_draftkings_lineup_" + str(game_date.month) + "_" + str(game_date.day) + "_" + str(game_date.year) + ".csv")
    return csv_return_dfs, csv_return_projections, lineup_return, lineups_100, players_in