"""
Script to analyse energy consumption data.

Compiled data can be found in compiled_data.csv. The script works by first reading in this data.

Run `pip install -r requirements.txt` installs all necessary dependencies for this project.

There are 4 components to the analysis in this script:

1. Loading and cleaning the compiled dataset 
2. Plotting the energy consumption per person graphically as a map
3. A brief exploratory analyis of the dataset, generating some simple plots
4. A causal analysis on the drivers of energy consumption, using a Bayesian linear regression

"""

############# IMPORTS AND SET UP ##############################
import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
logging.info("Importing relevant python packages") 
import numpy as np
import pandas as pd
import geopandas as gpd


############# READ / CLEAN DATASET ############################
logging.info("Reading in data")
try:
    df = pd.read_csv("compiled_data.csv")
    logging.info(f"Compiled dataset read with shape {df.shape}")
except OSError as e:
    logging.error(f"No file found, try running compile_data.py first. \n{e}")


# data cleaning
df["politically_green"] = [1 if x == True else 0 for x in df.politically_green]

############# PLOTTING ENERGY CONSUMPTION ON A MAP ############


############# EXPLORATORY ANALYSIS ############################


############# CAUSAL ANALYSIS #################################
