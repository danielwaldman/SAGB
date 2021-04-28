from hardcoded import *
import pandas as pd
from os import path
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import datetime as datetime
from datetime import datetime, timedelta
from dateutil.parser import parse

def scrape_bbref_data():
	"""Scrapes data from basketball-reference for months and years in above months_and_years.
	Currently scrapes from November 2016 to April 2021."""
	for m, y in months_and_years: #for current month, scrape up to today's date
		if m == datetime.today().month and y == CURRENT_YEAR:
			for d in range(1, datetime.today().day):
				file_name = "./AllCSVs/{0}_{1}_{2}_box_scores.csv".format(m, d, y)
				if path.exists(file_name):
					continue
				client.player_box_scores(day=d, month=m, year=y, output_type=OutputType.CSV, output_file_path=file_name)
		elif m == 2 and y % 4 != 0:
			for d in range(1, 29):
				file_name = "./AllCSVs/{0}_{1}_{2}_box_scores.csv".format(m, d, y)
				if path.exists(file_name):
					continue
				client.player_box_scores(day=d, month=m, year=y, output_type=OutputType.CSV, output_file_path=file_name)
		elif m == 2 and y % 4 == 0:
			for d in range(1, 30):
				file_name = "./AllCSVs/{0}_{1}_{2}_box_scores.csv".format(m, d, y)
				if path.exists(file_name):
					continue
				client.player_box_scores(day=d, month=m, year=y, output_type=OutputType.CSV, output_file_path=file_name)
		elif m in [4, 9, 11]:
			for d in range(1, 31):
				file_name = "./AllCSVs/{0}_{1}_{2}_box_scores.csv".format(m, d, y)
				if path.exists(file_name):
					continue
				client.player_box_scores(day=d, month=m, year=y, output_type=OutputType.CSV, output_file_path=file_name)
		else:
			for d in range(1, 32):
				file_name = "./AllCSVs/{0}_{1}_{2}_box_scores.csv".format(m, d, y)
				if path.exists(file_name):
					continue
				client.player_box_scores(day=d, month=m, year=y, output_type=OutputType.CSV, output_file_path=file_name)

def scrape_team_box_scores():
	"""Scrapes team box scores from basketball-reference for months and years in above months_and_years.
	Currently scrapes from November 2016 to April 2021."""
	for m, y in months_and_years: #for current month, scrape up to today's date
		if m == datetime.today().month and y == CURRENT_YEAR:
			for d in range(1, datetime.today().day):
				file_name = "./TeamBoxScores/{0}_{1}_{2}_team_box_scores.csv".format(m, d, y)
				if path.exists(file_name):
					continue
				client.team_box_scores(day=d, month=m, year=y, output_type=OutputType.CSV, output_file_path=file_name)
		elif m == 2 and y % 4 != 0:
			for d in range(1, 29):
				file_name = "./TeamBoxScores/{0}_{1}_{2}_team_box_scores.csv".format(m, d, y)
				if path.exists(file_name):
					continue
				client.team_box_scores(day=d, month=m, year=y, output_type=OutputType.CSV, output_file_path=file_name)
		elif m == 2 and y % 4 == 0:
			for d in range(1, 30):
				file_name = "./TeamBoxScores/{0}_{1}_{2}_team_box_scores.csv".format(m, d, y)
				if path.exists(file_name):
					continue
				client.team_box_scores(day=d, month=m, year=y, output_type=OutputType.CSV, output_file_path=file_name)
		elif m in [4, 9, 11]:
			for d in range(1, 31):
				file_name = "./TeamBoxScores/{0}_{1}_{2}_team_box_scores.csv".format(m, d, y)
				if path.exists(file_name):
					continue
				client.team_box_scores(day=d, month=m, year=y, output_type=OutputType.CSV, output_file_path=file_name)
		else:
			for d in range(1, 32):
				file_name = "./TeamBoxScores/{0}_{1}_{2}_team_box_scores.csv".format(m, d, y)
				if path.exists(file_name):
					continue
				client.team_box_scores(day=d, month=m, year=y, output_type=OutputType.CSV, output_file_path=file_name)

	def add_to_table(table, y, m, d):
		file_name_to_add = "./TeamBoxScores/{0}_{1}_{2}_team_box_scores.csv".format(m, d, y)
		temp = pd.read_csv(file_name_to_add)
		days = [d] * len(temp)
		months = [m] * len(temp)
		years = [y] * len(temp)
		temp["Day"] = days
		temp["Month"] = months
		temp["Year"] = years
		table = table.append(temp)
		return table
	
	teambox_columns = ["team", "minutes_played", "made_field_goals", "attempted_field_goals", 
			   "made_three_point_field_goals", "attempted_three_point_field_goals", "made_free_throws",
			  "attempted_free_throws", "offensive_rebounds", "defensive_rebounds", "assists",
			  "steals", "blocks", "turnovers", "personal_fouls", "Date"]
	
	all_box_scores = pd.DataFrame(columns = teambox_columns)
	for m, y in months_and_years:
		if m == datetime.today().month and y == CURRENT_YEAR:
			for d in range(1, datetime.today().day):
				all_box_scores = add_to_table(all_box_scores, y = y, m = m, d = d)
		elif m == 2:
			if y % 4 == 0:
				for d in range(1, 30):
					all_box_scores = add_to_table(all_box_scores, y = y, m = m, d = d)
			else:
				for d in range(1, 29):
					all_box_scores = add_to_table(all_box_scores, y = y, m = m, d = d)

		elif m in [4, 9, 11]:
			for d in range(1, 31):
				all_box_scores = add_to_table(all_box_scores, y = y, m = m, d = d)
		
		else:
			for d in range(1, 32):
				all_box_scores = add_to_table(all_box_scores, y = y, m = m, d = d)
				
	all_box_scores.reset_index()
	all_box_scores.to_csv("./TeamBoxScores/all_box_scores.csv")


def load_bbref_data():
	"""Loads data scraped above for months_and_years and returns it all in one table."""
	all_tables = []
	for m, y in months_and_years:
		if m == datetime.today().month and y == CURRENT_YEAR:
			for d in range(1, datetime.today().day):
				file_name = "./AllCSVs/{0}_{1}_{2}_box_scores.csv".format(m, d, y)
				table  = pd.read_csv(file_name)
				date = datetime(y, m, d)
				dates = [date] * len(table)
				table["Date"] = dates
				all_tables.append(table)
		elif m == 2:
			if y % 4 == 0:
				for d in range(1, 30):
					file_name = "./AllCSVs/{0}_{1}_{2}_box_scores.csv".format(m, d, y)
					table  = pd.read_csv(file_name)
					date = datetime(y, m, d)
					dates = [date] * len(table)
					table["Date"] = dates
					all_tables.append(table)
			else:
				for d in range(1, 29):
					file_name = "./AllCSVs/{0}_{1}_{2}_box_scores.csv".format(m, d, y)
					table  = pd.read_csv(file_name)
					date = datetime(y, m, d)
					dates = [date] * len(table)
					table["Date"] = dates
					all_tables.append(table)

		elif m in [4, 9, 11]:
			for d in range(1, 31):
				file_name = "./AllCSVs/{0}_{1}_{2}_box_scores.csv".format(m, d, y)
				table  = pd.read_csv(file_name)
				date = datetime(y, m, d)
				dates = [date] * len(table)
				table["Date"] = dates
				all_tables.append(table)
		
		else:
			for d in range(1, 32):
				file_name = "./AllCSVs/{0}_{1}_{2}_box_scores.csv".format(m, d, y)
				table  = pd.read_csv(file_name)
				date = datetime(y, m, d)
				dates = [date] * len(table)
				table["Date"] = dates
				all_tables.append(table)
	return all_tables


def write_bbref_data():
	"""Calls load_bbref_data() and writes it to all_games.csv."""
	all_tables = load_bbref_data()
	full_df = all_tables[0]
	for i in range(1, len(all_tables)):
		current_table = all_tables[i]
		full_df = full_df.append(current_table)
	full_df.to_csv("./OutputCSVs/all_games.csv")


def scrape_defensive_ratings():
	"""Scrapes Hashtag Basketball's NBA Defense vs Position for each season in the data 
	and writes it to OutputCSVs."""
	res = requests.get('https://hashtagbasketball.com/nba-defense-vs-position')
	soup = BeautifulSoup(res.text, 'lxml')
	pg, sg, sf, pf, c = [], [], [], [], []
	table = soup.find('table', attrs={'id':'ContentPlaceHolder1_GridView1'})
	for tr in table.find_all('tr'):
		td_list = tr.find_all('td')
		if tr.find('td') is not None:
			text = " ".join([i for i in td_list[1].text.split() if not i.isdigit()])
			if td_list[0].text == 'PG':
				pg.append(text)
			elif td_list[0].text == 'SG':
				sg.append(text)
			elif td_list[0].text == 'SF':
				sf.append(text)
			elif td_list[0].text == 'PF':
				pf.append(text)
			elif td_list[0].text == 'C':
				c.append(text)
	rank = list(range(1, 31))
	columns = ['Team', 'vs PG', 'vs SG', 'vs SF', 'vs PF', 'vs C']
	df_pg = pd.DataFrame({'Team':pg, 'vs PG':rank}).sort_values('Team')
	df_sg = pd.DataFrame({'Team':sg, 'vs SG':rank}).sort_values('Team')
	df_sf = pd.DataFrame({'Team':sf, 'vs SF':rank}).sort_values('Team')
	df_pf = pd.DataFrame({'Team':pf, 'vs PF':rank}).sort_values('Team')
	df_c = pd.DataFrame({'Team':c, 'vs C':rank}).sort_values('Team')
	a = pd.merge(df_pg, df_sg, on='Team')
	b = pd.merge(a, df_sf, on='Team')
	c = pd.merge(b, df_pf, on='Team')
	d = pd.merge(c, df_c, on='Team')
	df = d.reindex(columns=columns)
	fileoutput = "./OutputCSVs/team_def_vs_position_" + str(CURRENT_YEAR) + ".csv"
	df.to_csv(fileoutput, header=True, index=False)


