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
from netCDF4 import Dataset


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

logger.info("Tidying up the columns")
df = df[['Local Authority Name', 'Local Authority Code', 'MSOA Name',
       'Middle Layer Super Output Area (MSOA) Code', 'LSOA Name',
       'Lower Layer Super Output Area (LSOA) Code', 'coords',
       'pct_electric', 'Average Energy Consumption per Person (kWh)']].copy()

df.columns = ['LA_name', 'LA', 'MSOA_ame',
       'MSOA', 'LSOA_name',
       'LSOA', 'coords',
       'pct_electric', 'energy_consumption_per_person']

"""
COMPILING THE DATASET
---------------------
Do this in phases, one dataset at a time. 

1. Load in new source
2. Generate the relevant feature(s)
3. Merge with main "df" dataset
4. Move onto the next source!

"""

####################### CLIMATE / TEMPERATURE DATA ####################### 
logger.info("\n\nAdding climate / temperaure data from HADUK 60km grid dataset")
file_name = "data/tas_hadukgrid_uk_60km_ann_202101-202112.nc"
logger.info("Load temperaure measurement data and extract latitudes, longitudes, temperatures for each measurement")
climate_data = Dataset(file_name)
latitude = file_id.variables["latitude"][:,:]
longitude = file_id.variables["longitude"][:,:]
temps = file_id.variables["tas"][:,:]
logger.info(f"Loading each of the {len(temps[0])} temperature observations into a dictionary, removing null values") 
lats = [np.mean(x) for x in latitude] # take average of temp from each point in grid
longs = [np.mean(x) for x in longitude] 
ts = [np.mean(x) for x in temps[0]]
temp_dict = {co:t for co,t in zip(list(zip(lats, longs)), ts)}
temp_dict = {k: v for k, v in temp_dict.items() if v > 0} # some measurement points have null values

def find_closest_temp_measurement(this_point):
    """
    Find the nearest temperature measurement from the dataset.
    
    Parameters
    ----------
    this_point: tuple
        a (latitude, longitude) tuple
    
    Returns
    -------
    temperature: float
        the average temperaure from the nearest temperature measurement to "this_point" in 2021
    """
    return temp_dict[min(temp_dict.keys(), key=lambda x: geopy.distance.geodesic(this_point, x))]

logger.info("Generating tuples of (lat, long) in the main dataset, finding the nearest measurement by geodetic distance")
df["coords"] = [(lat, long) for lat, long in zip(df.Latitude, df.Longitude)]
df["temperature"] = [find_closest_temp_measurement(x) for x in df.coords]


####################### ENERGY COST DATA ###################################
logger.info("\n\nAdding energy cost data based on the electric vs gas usage of each LSOA")
df["pct_electric"] = df['Electricity Consumption (kWh)'] / dfA['Total Energy Consumption (kWh)']
logger.info(f"Compute estimate for relative energy cost by LSOA, assuming gas price of {GAS_PRICE_PER_KWH}p per kwh and electric price of {ELECTRIC_PRICE_PER_KWH}p per kwh")
df["energy_cost"] = [ELECTRIC_PRICE_PER_KWH * x + GAS_PRICE_PER_KWH * (1-x) for x in df["pct_electric"]]


####################### INCOME DATA ########################################
logger.info("\n\nAdding net income data (post housing costs) per household from ONS, provided by MSOA")
income_data = pd.read_csv("data/net_income_after_housing_costs.csv")
income_data = income_data[["MSOA code", "Net annual income after housing costs (£)"]].copy()
income_data.columns = ["MSOA", "net_income"]
df = df.merge(income_data, on="MSOA", how="left")


####################### ECONOMIC ACTIVITY DATA ##############################
logger.info("\n\nAdding dataset on economic activity from ONS, by local authority")
economic_activity = pd.read_csv("data/economic_activity.csv")
economic_activity = economic_activity[["Area code", "Economically active: \nIn employment \n(including full-time students), \n2021\n(percent)"]]
economic_activity.columns = ["LA", "pct_economically_active"]
df = df.merge(economic_activity, on="LA", how="left")


####################### HOME OCCUPANCY DATA ##################################
households = pd.read_csv("data/RM202-Household-Size-By-Number-Of-Rooms-2021-lsoa-ONS.csv")
households.rename(columns={"Lower layer Super Output Areas Code": "LSOA"}, inplace=True)
households["pct_home_occupancy"] = households["Household size (5 categories) Code"] / households["Number of rooms (Valuation Office Agency) (6 categories) Code"]
households["pct_home_occupancy_x_obs"] = households["pct_home_occupancy"] * households["Observation"]
households["home_size_x_obs"] = households["Number of rooms (Valuation Office Agency) (6 categories) Code"] * households["Observation"]
totals = households.groupby("LSOA")[["pct_home_occupancy_x_obs", "home_size_x_obs", "Observation"]].sum().reset_index()
totals["home_size"] = totals["home_size_x_obs"] / totals["Observation"]
totals["pct_home_occupancy"] = totals["pct_home_occupancy_x_obs"] / totals["Observation"]
totals = totals[["LSOA", "home_size", "pct_home_occupancy"]]
df = df.merge(totals, on="LSOA", how="left")

"""
WRITING RESULTS
---------------
"""