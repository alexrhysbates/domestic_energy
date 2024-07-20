"""
Script to compile energy consumption data into a single csv file.

Raw data can be found in raw_data.zip, extract before running this file. The script works by picking up the raw data from a local "raw/" folder.

Run `pip install -r requirements.txt` installs all necessary dependencies for this project.

General approach is to start with the "usmart" energy consumption dataset, and merge in new features from other datasets, joining on either the local area code, MSOA or LSOA

Features (and associated assumptions):
*
*
*
*

"""

######### IMPORTS #########
import logging
import numpy as np
import pandas as pd


#### LOADING THE DATA #####
energy_consumption_data = pd.read_csv("raw/LSOA Energy Consumption Data.csv")




#### FEATURE GENERATION ####

## COMPILING THE DATASET ##



#