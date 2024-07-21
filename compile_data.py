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

"""
IMPORTS AND SET UP
------------------
Load relevant python packages and set up a logger

"""
import logging
logger = logging.getLogger(__name__)

logger.info("Importing relevant python packages") 
import numpy as np
import pandas as pd
import geopy.distance


"""
ASSUMPTIONS
-----------
We need to set a few assumptions on energy prices
Taken from:
https://www.icaew.com/insights/viewpoints-on-the-news/2022/sept-2022/chart-of-the-week-energy-price-cap-update

"""
GAS_PRICE_PER_KWH = 3.3
ELECTRIC_PRICE_PER_KWH = 19.0


"""
LOADING THE DATA
----------------
Read in the main dataset
"""
logger.info("Loading main dataset on energy consumption")
df = pd.read_csv("data/LSOA Energy Consumption Data.csv")
logger.info(f"Data loaded, {len(df)} rows")


"""
COMPILING THE DATASET
---------------------
Do this in phases, one dataset at a time. 

1. Load in new source
2. Generate the relevant feature(s)
3. Merge with main "df" dataset
4. Move onto the next source!

"""



## WRITING TO CSV