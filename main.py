from scraping import *
from processing import *
from modeling import *
from predicting import *
from hardcoded import *


def main():
	# ---------- SCRAPING ----------
	scrape_bbref_data()
	scrape_team_box_scores()
	write_bbref_data()
	scrape_defensive_ratings()
	print("Done with scraping! Hopefully that didn't take too long.")

	# ---------- PROCESSING ----------
	attach_team_stats()
	attach_b2b_indicators()
	print("Added some numbers. This next part will take a while.")

	# ---------- PREDICTING ----------
	predict_unplayed_games(datetime(2016, 11, 1), datetime(2021, 4, 26), retrain = True, pretrain_inputs = True)

	preds = pd.read_csv("./AllCSVs/predictions_for_04_26_2021_unplayed.csv")#.append(pd.read_csv("./AllCSVs/predictions_for_09_20_2020_unplayed.csv"))
	#preds.index = range(0, len(preds))

	fd_csv = "./AllCSVs/FanDuel-NBA-2021-04-26-players-list.csv"
	
	dk_csv = "./AllCSVs/DKSalaries_04262021.csv"

	players_out = ["Victor Oladipo",
			   "Shai Gilgeous-Alexander",
			   "Eric Paschall",
			   "Alec Burks",
			   "Michael Carter-Williams",
			   "Zach LaVine",
			   "Al Horford",
			   "Kris Dunn",
			   "Kevin Durant",
			   "Jamal Murray",
			   "Danilo Gallinari",
			   "Evan Fournier",
			   "Mike Muscala",
			   "Malik Monk",
			   "LaMelo Ball",
			   "Eric Gordon",
			   "LeBron James", 
			   "Markelle Fultz", 
			   "Spencer Dinwiddie", 
			   "Anthony Davis", 
			   "T.J. Warren", 
			   "Marquese Chriss",
			   "Jonathan Isaac", 
			   "Klay Thompson", 
			   "Jabari Parker"]

	optimal_lineup_fanduel_games(datetime(2021, 4, 26), preds, fd_csv, players_out)[2]
	optimal_lineup_draftkings_games(datetime(2021, 4, 26), preds, dk_csv, players_out)[2]

if __name__ == "__main__":
	main()
