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
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns


############# READ / CLEAN DATASET ############################
logging.info("Reading in data")
try:
    df = pd.read_csv("compiled_data.csv")
    logging.info(f"Compiled dataset read with shape {df.shape}")
except OSError as e:
    logging.error(f"No file found, try running compile_data.py first. \n{e}")


logging.info("Cleaning up columns with any null values or incorrect dtypes")
df["politically_green"] = [1 if x == True else 0 for x in df.politically_green]
df["net_income"] = [int(x.replace(",","").strip()) for x in df.net_income]


############# PLOTTING ENERGY CONSUMPTION ON A MAP ############
logging.info("Using geopandas to load and plot energy consumption per capita on a map")
logging.info("Read shapefile as a dataframe and tidy up columns")
las = gpd.GeoDataFrame.from_file("data/LAD_DEC_2021_UK_BFC.shp")
las.rename(columns={"LAD21CD":"LA"}, inplace=True)
logging.info("Join shapefile with energy data, averaged by local authority, and plot")
df_energy = df.groupby("LA")["energy_consumption_per_person"].mean().reset_index()
las = las.merge(df_energy, on="LA", how="left")
las.plot(column="energy_consumption_per_person", cmap="OrRd", edgecolor="k", legend=True)
logging.info("Saving plot to a local .png")
plt.savefig('energy_consumption_per_person_by_uk_local_authority.png')


############# EXPLORATORY ANALYSIS ############################
corr_data = df.drop(columns=['LA', 'MSOA', 'LSOA'], axis=1).corr()
corr_heatmap = sns.heatmap(corr_data, cmap="YlGnBu", annot=True) 
plt.savefig("correlations_of_features.png")

############# CAUSAL ANALYSIS #################################
