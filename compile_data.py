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
logger.info(f"Check dataset shape: {df.shape}")

####################### ENERGY COST DATA ###################################
logger.info("\n\nAdding energy cost data based on the electric vs gas usage of each LSOA")
df["pct_electric"] = df['Electricity Consumption (kWh)'] / dfA['Total Energy Consumption (kWh)']
logger.info(f"Compute estimate for relative energy cost by LSOA, assuming gas price of {GAS_PRICE_PER_KWH}p per kwh and electric price of {ELECTRIC_PRICE_PER_KWH}p per kwh")
df["energy_cost"] = [ELECTRIC_PRICE_PER_KWH * x + GAS_PRICE_PER_KWH * (1-x) for x in df["pct_electric"]]
logger.info(f"Check dataset shape: {df.shape}")

####################### INCOME DATA ########################################
logger.info("\n\nAdding net income data (post housing costs) per household from ONS, provided by MSOA")
logger.info("Read in csv and tidy up columns")
income_data = pd.read_csv("data/net_income_after_housing_costs.csv")
income_data = income_data[["MSOA code", "Net annual income after housing costs (£)"]].copy()
income_data.columns = ["MSOA", "net_income"]
logger.info("Merge with main dataset")
df = df.merge(income_data, on="MSOA", how="left")
logger.info(f"Check dataset shape: {df.shape}")


####################### ECONOMIC ACTIVITY DATA ##############################
logger.info("\n\nAdding dataset on economic activity from ONS, by local authority")
logger.info("Read in csv and tidy up columns")
economic_activity = pd.read_csv("data/economic_activity.csv")
economic_activity = economic_activity[["Area code", "Economically active: \nIn employment \n(including full-time students), \n2021\n(percent)"]]
economic_activity.columns = ["LA", "pct_economically_active"]
logger.info("Merge with main dataset")
df = df.merge(economic_activity, on="LA", how="left")
logger.info(f"Check dataset shape: {df.shape}")

####################### HOME OCCUPANCY DATA ##################################
logger.info("\n\nAdding dataset on home size and household occupancy from ONS for 2021. Features by LSOA")
logger.info("Read in csv and tidy up columns")
households = pd.read_csv("data/RM202-Household-Size-By-Number-Of-Rooms-2021-lsoa-ONS.csv")
households.rename(columns={"Lower layer Super Output Areas Code": "LSOA"}, inplace=True)

logger.info("Compute the percentage of home occupancy and multiply by no. of observations of that occupancy")
households["pct_home_occupancy"] = households["Household size (5 categories) Code"] / households["Number of rooms (Valuation Office Agency) (6 categories) Code"]
households["pct_home_occupancy_x_obs"] = households["pct_home_occupancy"] * households["Observation"]

logger.info("Compute the size of each home multiplied by the no. of observations of that size.")
households["home_size_x_obs"] = households["Number of rooms (Valuation Office Agency) (6 categories) Code"] * households["Observation"]

logger.info("Sum each of these computed fields for each LSOA and divide by the total number of homes in that LSOA")
totals = households.groupby("LSOA")[["pct_home_occupancy_x_obs", "home_size_x_obs", "Observation"]].sum().reset_index()
totals["home_size"] = totals["home_size_x_obs"] / totals["Observation"]
totals["pct_home_occupancy"] = totals["pct_home_occupancy_x_obs"] / totals["Observation"]
logger.info(f"Mean home size of {totals.home.size.mean()}. Mean occupancy % of {totals.pct_home_occupancy.mean()}")

logger.info("Tidy up columns and merge with main dataset")
totals = totals[["LSOA", "home_size", "pct_home_occupancy"]]
df = df.merge(totals, on="LSOA", how="left")
logger.info(f"Check dataset shape: {df.shape}")

####################### BUILDING TYPE DATA ##################################
logger.info("\n\nAdding data on building type from gov.uk council tax dataset on stock of properties for 2021") 
building_type = pd.read_csv("data/CTSOP_3_1_2021.csv")
logger.info("Filtering down to LSOA only and all council tax bands, tidying up columns and replacing dodgy characters")
building_type = building_type[(building_type.geography == "LSOA") & (building_type.band == "All")]
building_type = building_type[["ecode", "bungalow_total", "flat_mais_total", "house_terraced_total",
                         "house_semi_total", "house_detached_total", "all_properties"]]
building_type = building_type.replace("-","0")

logger.info("Define the number of 'exposed surfaces' per building type - assumption that flats are more energy-efficient for this reason")
exposed_surfaces_per_type = {
    "bungalow_total": 5,
    "flat_mais_total": 2,
    "house_terraced_total": 3,
    "house_semi_total": 4,
    "house_detached_total": 5
}

logger.info("Convert the building type count columns to integers")
building_type[list(exposed_surfaces_per_type.keys())] = building_type[exposed_surfaces_per_type.keys()].astype(int)

logger.info("Multiply building type count columns by the number of exposed surfaces to get totals")
total_exposed_surfaces = building_type[list(exposed_surfaces_per_type.keys())].mul(exposed_surfaces_per_type).sum(axis=1)
logger.info("Compute the average number of exposed surfaces for properties in that LSOA")
building_type["home_exposed_surfaces"]  = [x / int(y) for x,y in zip(total_exposed_surfaces,building_type["all_properties"])]

logger.info("Tidy up columns and merge with main dataset")
building_type = building_type[["ecode", "home_exposed_surfaces"]]
building_type.columns = ["LSOA", "home_exposed_surfaces"]
df = df.merge(building_type, on="LSOA", how="left")
logger.info(f"Check dataset shape: {df.shape}")

####################### BUILDING AGE DATA ###################################


logger.info(f"Check dataset shape: {df.shape}")

"""
WRITING RESULTS
---------------
"""