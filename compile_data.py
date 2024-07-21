"""
Script to compile energy consumption data into a single csv file.

Raw data can be found in raw_data.zip, extract before running this file. The script works by picking up the raw data from a local "data/" folder.

Run `pip install -r requirements.txt` installs all necessary dependencies for this project.

General approach is to start with the "usmart" energy consumption dataset, and merge in new features from other datasets, joining on either the local area code, MSOA or LSOA. Given we need to join multiple datasets together, it makes sense to work in pandas, and make use of the DataFrame.merge() function!

The usmart dataset was released in 2021 - we will try to use datasets that match this assumed effective date / period - handy that there was a UK census in 2021!

Target:
* energy_consumption_per_person - from usmart dataset, by LSOA

Features (and associated sources / definitions / assumptions):
* net_income
* energy_cost
* temperature
* politically_green
* pct_economically_active
* home_age
* home_size
* pct_home_occupancy
* pct_houses_detatched

"""

######### IMPORTS #########
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import geopy.distance


#### LOADING THE DATA #####
energy_consumption_data = pd.read_csv("raw/LSOA Energy Consumption Data.csv")

## COMPILING THE DATASET ##



#