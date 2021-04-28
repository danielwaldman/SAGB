CURRENT_YEAR = 2021
months_and_years = [(4, 2021), (3, 2021), (2, 2021), (1, 2021), (12, 2020), (8, 2020), (11, 2019), (12, 2019), (1, 2020), (2, 2020), (3, 2020), (7, 2020), (11, 2018), (12, 2018), (1, 2019), (2, 2019), (3, 2019), (4, 2019), (11, 2017), (12, 2017), (1, 2018), (2, 2018), (3, 2018), (4, 2018), (11, 2016), (12, 2016), (1, 2017), (2, 2017), (3, 2017), (4, 2017)]
input_statistics = ["name", "team", "date", "location", "opponent", "made_field_goals", "made_two_point_field_goals", "attempted_two_point_field_goals", "attempted_field_goals", "made_three_point_field_goals", "attempted_three_point_field_goals", "attempted_free_throws", "made_free_throws", "offensive_rebounds", "defensive_rebounds", "assists", "blocks", "turnovers", "steals", "seconds_played", "Opponent Defensive Rating", "Opponent Turnover %", 'Team Defensive Rating', 'Team Pace', 'Team Turnover %', 'Opponent Pace']
cols_to_average = ["seconds_played", "made_field_goals", "attempted_field_goals", "made_three_point_field_goals", "attempted_three_point_field_goals", "attempted_free_throws", "made_free_throws", "offensive_rebounds", "defensive_rebounds", "assists", "blocks", "turnovers", "steals", "game_score", "Opponent Defensive Rating", "Opponent Turnover %", 'Team Defensive Rating', 'Team Pace', 'Team Turnover %', 'Opponent Pace', "attempted_two_point_field_goals", "made_two_point_field_goals", "is_win"]
output_statistics = ["name", "team", "date", "location", "opponent", "minutes", "made_two_point_field_goals", "made_three_point_field_goals", "made_free_throws", "rebounds", "assists", "blocks", "steals", "turnovers", "recent_average", "10_game_average", "3_game_average", "10_3_ratio", "10_3_difference", "hot", "cold", "fantasy_points"]
all_abbrv = {'ATLANTA HAWKS':'ATL', 'BOSTON CELTICS':'BOS', 'BROOKLYN NETS':'BRO', 'CHARLOTTE HORNETS':'CHA', 'CHICAGO BULLS':'CHI', 'CLEVELAND CAVALIERS':'CLE', 'DALLAS MAVERICKS':'DAL',
						'DENVER NUGGETS':'DEN', 'DETROIT PISTONS':'DET', 'GOLDEN STATE WARRIORS':'GSW', 'HOUSTON ROCKETS':'HOU', 'INDIANA PACERS':'IND', 'LOS ANGELES CLIPPERS':'LAC', 'LOS ANGELES LAKERS':'LAL',
						'MEMPHIS GRIZZLIES':'MEM', 'MIAMI HEAT':'MIA', 'MILWAUKEE BUCKS':'MIL', 'MINNESOTA TIMBERWOLVES':'MIN', 'NEW ORLEANS PELICANS':'NOP', 'NEW YORK KNICKS':'NYK', 'OKLAHOMA CITY THUNDER':'OKL', 'ORLANDO MAGIC':'ORL',
						'PHILADELPHIA 76ERS':'PHI', 'PHOENIX SUNS':'PHX', 'PORTLAND TRAIL BLAZERS':'POR', 'SACRAMENTO KINGS':'SAC', 'SAN ANTONIO SPURS':'SAS', 'TORONTO RAPTORS':'TOR', 'UTAH JAZZ':'UTA', 'WASHINGTON WIZARDS':'WAS'}
betting_dictionary = {'ATLANTA HAWKS':'Hawks', 'BOSTON CELTICS':'Celtics', 'BROOKLYN NETS':'Nets', 'CHARLOTTE HORNETS':'Hornets', 'CHICAGO BULLS':'Bulls', 'CLEVELAND CAVALIERS':'Cavaliers', 'DALLAS MAVERICKS':'Mavericks',
						'DENVER NUGGETS':'Nuggets', 'DETROIT PISTONS':'Pistons', 'GOLDEN STATE WARRIORS':'Warriors', 'HOUSTON ROCKETS':'Rockets', 'INDIANA PACERS':'Pacers', 'LOS ANGELES CLIPPERS':'Clippers', 'LOS ANGELES LAKERS':'Lakers',
						'MEMPHIS GRIZZLIES':'Grizzlies', 'MIAMI HEAT':'Heat', 'MILWAUKEE BUCKS':'Bucks', 'MINNESOTA TIMBERWOLVES':'Timberwolves', 'NEW ORLEANS PELICANS':'Pelicans', 'NEW YORK KNICKS':'Knicks', 'OKLAHOMA CITY THUNDER':'Thunder', 'ORLANDO MAGIC':'Magic',
						'PHILADELPHIA 76ERS':'Seventysixers', 'PHOENIX SUNS':'Suns', 'PORTLAND TRAIL BLAZERS':'Trailblazers', 'SACRAMENTO KINGS':'Kings', 'SAN ANTONIO SPURS':'Spurs', 'TORONTO RAPTORS':'Raptors', 'UTAH JAZZ':'Jazz', 'WASHINGTON WIZARDS':'Wizards'}

# Maps of names from those in the data to the formats Fanduel and Draftkings have.
# These only include players whose teams were in the bubble, and of course are not yet updated
# for the 2020-21 season. Additional work is required to make these lists complete.

difficult_names_map_fanduel = {"Luka Dončić": "Luka Doncic", 
											 "Luka Šamanić": "Luka Samanic", 
											 "Kristaps Porziņģis": "Kristaps Porzingis", 
											 "Nikola Vučević": "Nikola Vucevic",
											 "Jonas Valančiūnas": "Jonas Valanciunas",
											 "Bogdan Bogdanović": "Bogdan Bogdanovic",
											 "Dario Šarić": "Dario Saric",
											 "Timothé Luwawu-Cabarrot": "Timothe Luwawu-Cabarrot",
											 "Džanan Musa": "Dzanan Musa",
												"Dāvis Bertāns": "Davis Bertans",
												"Boban Marjanović": "Boban Marjanovic",
												"Ersan İlyasova": "Ersan Ilyasova",
												"Anžejs Pasečņiks": "Anzejs Pasecniks",
											 "Bojan Bogdanović": "Bojan Bogdanovic",
												"Nicolò Melli": "Nicolo Melli",
												"Nikola Jokić": "Nikola Jokic",
												"Jusuf Nurkić": "Jusuf Nurkic",
												"Goran Dragić": "Goran Dragic",
												"Dennis Schröder" :"Dennis Schroder",
											 "Gary Payton": "Gary Payton II",
											 "Mohamed Bamba": "Mo Bamba",
											 "Wesley Iwundu": "Wes Iwundu",
												"J.J. Redick": "JJ Redick",
												"B.J. Johnson": "BJ Johnson"} #Check this for August 1

difficult_names_map_draftkings = {"Luka Dončić": "Luka Doncic", 
											 "Luka Šamanić": "Luka Samanic", 
											 "Kristaps Porziņģis": "Kristaps Porzingis", 
											 "Nikola Vučević": "Nikola Vucevic",
											 "Jonas Valančiūnas": "Jonas Valanciunas",
											 "Bogdan Bogdanović": "Bogdan Bogdanovic",
											 "Dario Šarić": "Dario Saric",
											 "Timothé Luwawu-Cabarrot": "Timothe Luwawu-Cabarrot",
											 "Džanan Musa": "Dzanan Musa",
												"Boban Marjanović": "Boban Marjanovic",
												"Ersan İlyasova": "Ersan Ilyasova",
												"Anžejs Pasečņiks": "Anzejs Pasecniks",
											 "Bojan Bogdanović": "Bojan Bogdanovic",
												"Dāvis Bertāns": "Davis Bertans",
												"Nicolò Melli": "Nicolo Melli",
												"Nikola Jokić": "Nikola Jokic",
												"Jusuf Nurkić": "Jusuf Nurkic",
												"Goran Dragić": "Goran Dragic",
												"Dennis Schröder" :"Dennis Schroder",
											 "Gary Payton": "Gary Payton II",
											 "Frank Mason": "Frank Mason III",
											 "Marvin Bagley": "Marvin Bagley III",
											 "James Ennis": "James Ennis III",
											 "Harry Giles": "Harry Giles III",
												"Lonnie Walker": "Lonnie Walker IV",
											 "Mohamed Bamba": "Mo Bamba",
											 "Wesley Iwundu": "Wes Iwundu",
												"J.J. Redick": "JJ Redick",
												"B.J. Johnson": "BJ Johnson",
												"Melvin Frazier": "Melvin Frazier Jr.",
												"Gary Trent": "Gary Trent Jr.",
												"Danuel House": "Danuel House Jr.",
												"Tim Hardaway": "Tim Hardaway Jr.",
												"Jaren Jackson": "Jaren Jackson Jr.",
												"Kelly Oubre": "Kelly Oubre Jr.",
												"Troy Brown": "Troy Brown Jr.",
												"Marcus Morris": "Marcus Morris Sr."} #Check this for August 1

# Incomplete list of players whose names the optimal_lineup function had difficulty handling.
# If the optimal lineup is one player short, that player may belong in this dictionary.


punctuation_names = {"Kentavious Caldwell Pope": "Kentavious Caldwell-Pope",
										"Marcus Morris Sr.": "Marcus Morris",
										"Shai Gilgeous Alexander": "Shai Gilgeous-Alexander",
										"Al Farouq Aminu": "Al-Farouq Aminu",
										"Naz Mitrou Long": "Naz Mitrou-Long",
										"Talen Horton Tucker": "Talen Horton-Tucker",
										"Willie Cauley Stein": "Willie Cauley-Stein",
										"Karl Anthony Towns": "Karl-Anthony Towns",
										"Timothe Luwawu Cabarrot": "Timothe Luwawu-Cabarrot",
										"Troy Brown Jr.": "Troy Brown",
										"Danuel House Jr.": "Danuel House",
										"Tim Hardaway Jr.": "Tim Hardaway",
										"Kelly Oubre Jr.": "Kelly Oubre",
										"Dorian Finney Smith": "Dorian Finney-Smith",
										"Juan Toscano Anderson": "Juan Toscano-Anderson",
										"Michael Carter Williams": "Michael Carter-Williams",
										"Nickeil Alexander Walker": "Nickeil Alexander-Walker"}
fd_to_dk = {"James Ennis": "James Ennis III",
					 "Gary Trent": "Gary Trent Jr.",
					 "Marcus Morris": "Marcus Morris Sr.",
					 "Tim Hardaway": "Tim Hardaway Jr."}