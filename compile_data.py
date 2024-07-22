"""
Script to compile energy consumption data into a single csv file.

Raw data can be found in raw_data.zip, extract before running this file. The script works by picking up the raw data from a local "data/" folder.

Run `pip install -r requirements.txt` installs all necessary dependencies for this project.

General approach is to start with the "usmart" energy consumption dataset, and merge in new features from other datasets, joining on either the local area code, MSOA or LSOA. Given we need to join multiple datasets together, it makes sense to work in pandas, and make use of the DataFrame.merge() function!

The usmart dataset was released in 2021 - we will try to use datasets that match this assumed effective date / period - handy that there was a UK census in 2021!

Target:
* energy_consumption_per_person - from usmart dataset, by LSOA

Features (and associated sources / definitions / assumptions):
* temperature - from HAD 60km grid temp measurement data. Link LSOA to closest avg temp at closest measurement point
* energy_cost - taken from energy cap data, assume driven by relative use of electric vs gas within LSOA 
* net_income - income after housing costs by MSOA, taken from ONS
* politically_green - does local authority have > 10% Green party council? If so, assume eco-aware population
* pct_economically_active - from ONS data by local authority, what % of people are in full time employment or students
* home_size - average number of bedrooms in homes within LSOA, from Council Tax dataset
* pct_home_occupancy - pct of bedrooms occupied - no. of occupants / home_size, also from Council Tax dataset
* home_exposed_surfaces - derived from building type Council Tax dataset, detatched homes have 5 exposed surfaces, semis have 4 etc. Average for that LSOA.
* home_age - average age of home in the LSOA, from same Council Tax dataset

"""

"""
IMPORTS AND SET UP
------------------
Load relevant python packages and set up a logger

"""
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
logging.info("Importing relevant python packages")
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

Also set assumption for % of green vote that we define as making an area "green politically"
"""
GAS_PRICE_PER_KWH = 3.3
ELECTRIC_PRICE_PER_KWH = 19.0
POLITICALLY_GREEN_THRESHOLD = 0.1

"""
LOADING THE DATA
----------------
Read in the main dataset
"""
logging.info("\n\nLoading main dataset on energy consumption")
df = pd.read_csv("data/LSOA Energy Consumption Data.csv")
logging.info(f"Data loaded, {len(df)} rows")

logging.info("Tidying up the columns")
df = df[
    [
        "Local Authority Name",
        "Local Authority Code",
        "MSOA Name",
        "Middle Layer Super Output Area (MSOA) Code",
        "LSOA Name",
        "Lower Layer Super Output Area (LSOA) Code",
        "Latitude",
        "Longitude",
        "Electricity Consumption (kWh)",
        "Total Energy Consumption (kWh)",
        "Average Energy Consumption per Person (kWh)",
    ]
].copy()

df.columns = [
    "LA_name",
    "LA",
    "MSOA_name",
    "MSOA",
    "LSOA_name",
    "LSOA",
    "Latitude",
    "Longitude",
    "elec_consumption",
    "total_consumption",
    "energy_consumption_per_person",
]

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
logging.info("\n\nAdding climate / temperaure data from HADUK 60km grid dataset")
file_name = "data/tas_hadukgrid_uk_60km_ann_202101-202112.nc"
logging.info(
    "Load temperaure measurement data and extract latitudes, longitudes, temperatures for each measurement"
)
climate_data = Dataset(file_name)
latitude = climate_data.variables["latitude"][:, :]
longitude = climate_data.variables["longitude"][:, :]
temps = climate_data.variables["tas"][:, :]
logging.info(
    f"Loading each of the {len(temps[0])} temperature observations into a dictionary, removing null values"
)
lats = [np.mean(x) for x in latitude]  # take average of temp from each point in grid
longs = [np.mean(x) for x in longitude]
ts = [np.mean(x) for x in temps[0]]
temp_dict = {coord: t for coord, t in zip(list(zip(lats, longs)), ts)}
temp_dict = {
    k: v for k, v in temp_dict.items() if v > 0
} 


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
    return temp_dict[
        min(temp_dict.keys(), key=lambda x: geopy.distance.geodesic(this_point, x))
    ]


logging.info(
    "Generating tuples of (lat, long) in the main dataset, finding the nearest temperature measurement by geodetic distance"
)
df["coords"] = [(lat, long) for lat, long in zip(df.Latitude, df.Longitude)]
df["temperature"] = [find_closest_temp_measurement(x) for x in df.coords]
logging.info(f"Check dataset shape: {df.shape}")

####################### ENERGY COST DATA ###################################
logging.info(
    "\n\nAdding energy cost data based on the electric vs gas usage of each LSOA"
)
df["pct_electric"] = df["elec_consumption"] / df["total_consumption"]
logging.info(
    f"Compute estimate for relative energy cost by LSOA, assuming gas price of {GAS_PRICE_PER_KWH}p per kwh and electric price of {ELECTRIC_PRICE_PER_KWH}p per kwh"
)
df["energy_cost"] = [
    ELECTRIC_PRICE_PER_KWH * x + GAS_PRICE_PER_KWH * (1 - x) for x in df["pct_electric"]
]
logging.info(f"Check dataset shape: {df.shape}")

####################### INCOME DATA ########################################
logging.info(
    "\n\nAdding net income data (post housing costs) per household from ONS, provided by MSOA"
)
logging.info("Read in csv and tidy up columns")
income_data = pd.read_csv("data/net_income_after_housing_costs.csv")
income_data = income_data[
    ["MSOA code", "Net annual income after housing costs (£)"]
].copy()
income_data.columns = ["MSOA", "net_income"]
logging.info("Merge with main dataset")
df = df.merge(income_data, on="MSOA", how="left")
logging.info(f"Check dataset shape: {df.shape}")

####################### ADD VOTING DATA ####################################
logging.info(
    "\n\nAdd in voting data from the 2021 local elections, find local authorities with high pct of Green vote"
)
voting_data = pd.read_csv("data/CBP09228_detailed_results_England_elections.csv")
logging.info("Find % of vote that is green, and compute which LAs exceed threshold")
voting_data["pct_green"] = voting_data["Green"] / voting_data["Total"]
voting_data["green_council"] = voting_data["pct_green"] >= POLITICALLY_GREEN_THRESHOLD
logging.info("Tidy up columns and merge with main dataset")
voting_data = voting_data[["ONS code", "green_council"]].copy()
voting_data.columns = ["LA", "politically_green"]
df = df.merge(voting_data, on="LA", how="left")

####################### ECONOMIC ACTIVITY DATA ##############################
logging.info("\n\nAdding dataset on economic activity from ONS, by local authority")
logging.info("Read in csv and tidy up columns")
economic_activity = pd.read_csv("data/economic_activity.csv")
economic_activity = economic_activity[
    [
        "Area code",
        "Economically active: \nIn employment \n(including full-time students), \n2021\n(percent)",
    ]
]
economic_activity.columns = ["LA", "pct_economically_active"]
logging.info("Merge with main dataset")
df = df.merge(economic_activity, on="LA", how="left")
logging.info(f"Check dataset shape: {df.shape}")

####################### HOME OCCUPANCY DATA ##################################
logging.info(
    "\n\nAdding dataset on home size and household occupancy from ONS for 2021. Features by LSOA"
)
logging.info("Read in csv and tidy up columns")
households = pd.read_csv("data/RM202-Household-Size-By-Number-Of-Rooms-2021-lsoa-ONS.csv")
households.rename(columns={"Lower layer Super Output Areas Code": "LSOA"}, inplace=True)

logging.info(
    "Compute the percentage of home occupancy and multiply by no. of observations of that occupancy"
)
households["pct_home_occupancy"] = (
    households["Household size (5 categories) Code"]
    / households["Number of rooms (Valuation Office Agency) (6 categories) Code"]
)
households["pct_home_occupancy_x_obs"] = (
    households["pct_home_occupancy"] * households["Observation"]
)

logging.info(
    "Compute the size of each home multiplied by the no. of observations of that size."
)
households["home_size_x_obs"] = (
    households["Number of rooms (Valuation Office Agency) (6 categories) Code"]
    * households["Observation"]
)

logging.info(
    "Sum each of these computed fields for each LSOA and divide by the total number of homes in that LSOA"
)
totals = (
    households.groupby("LSOA")[
        ["pct_home_occupancy_x_obs", "home_size_x_obs", "Observation"]
    ]
    .sum()
    .reset_index()
)
totals["home_size"] = totals["home_size_x_obs"] / totals["Observation"]
totals["pct_home_occupancy"] = totals["pct_home_occupancy_x_obs"] / totals["Observation"]
logging.info(
    f"Mean home size of {totals.home_size.mean()}. Mean occupancy % of {totals.pct_home_occupancy.mean()}"
)

logging.info("Tidy up columns and merge with main dataset")
totals = totals[["LSOA", "home_size", "pct_home_occupancy"]]
df = df.merge(totals, on="LSOA", how="left")
logging.info(f"Check dataset shape: {df.shape}")

####################### BUILDING TYPE DATA ##################################
logging.info(
    "\n\nAdding data on building type from gov.uk council tax dataset on stock of properties for 2021"
)
building_type = pd.read_csv("data/CTSOP_3_1_2021.csv")
logging.info(
    "Filtering down to LSOA only and all council tax bands, tidying up columns and replacing dodgy characters"
)
building_type = building_type[
    (building_type.geography == "LSOA") & (building_type.band == "All")
]
building_type = building_type[
    [
        "ecode",
        "bungalow_total",
        "flat_mais_total",
        "house_terraced_total",
        "house_semi_total",
        "house_detached_total",
        "all_properties",
    ]
]
building_type = building_type.replace("-", "0")

logging.info(
    "Define the number of 'exposed surfaces' per building type as a dictionary - assumption that flats are more energy-efficient for this reason"
)
exposed_surfaces_per_type = {
    "bungalow_total": 5,
    "flat_mais_total": 2,
    "house_terraced_total": 3,
    "house_semi_total": 4,
    "house_detached_total": 5,
}

logging.info("Convert the building type count columns to integers")
building_type[list(exposed_surfaces_per_type.keys())] = building_type[
    exposed_surfaces_per_type.keys()
].astype(int)

logging.info(
    "Multiply building type count columns by the number of exposed surfaces to get totals"
)
total_exposed_surfaces = (
    building_type[list(exposed_surfaces_per_type.keys())]
    .mul(exposed_surfaces_per_type)
    .sum(axis=1)
)
logging.info("Compute the average number of exposed surfaces for properties in that LSOA")
building_type["home_exposed_surfaces"] = [
    x / int(y) for x, y in zip(total_exposed_surfaces, building_type["all_properties"])
]

logging.info("Tidy up columns and merge with main dataset")
building_type = building_type[["ecode", "home_exposed_surfaces"]]
building_type.columns = ["LSOA", "home_exposed_surfaces"]
df = df.merge(building_type, on="LSOA", how="left")
logging.info(f"Check dataset shape: {df.shape}")

####################### BUILDING AGE DATA ###################################
logging.info(
    "\n\nAdding in building age data from gov.uk council tax dataset on stock of properties for 2021"
)
building_age = pd.read_csv("data/CTSOP_4_1_2021.csv")
logging.info(
    "Filtering down to LSOA only and all council tax bands, tidying up columns and replacing weird characters"
)
building_age = building_age[
    (building_age.geography == "LSOA") & (building_age.band == "All")
]
building_age = building_age.replace("-", "0")

logging.info(
    "Building age given in bands - estimate a rough mid-point for bands where applicable, as a dictionary."
)
build_dates = {
    "bp_pre_1900": 1900,
    "bp_1900_1918": 1910,
    "bp_1919_1929": 1925,
    "bp_1930_1939": 1935,
    "bp_1945_1954": 1950,
    "bp_1955_1964": 1960,
    "bp_1965_1972": 1969,
    "bp_1973_1982": 1978,
    "bp_1983_1992": 1988,
    "bp_1993_1999": 1996,
    "bp_2000_2008": 2004,
    "bp_2009": 2009,
    "bp_2010": 2010,
    "bp_2011": 2011,
    "bp_2012": 2012,
    "bp_2013": 2013,
    "bp_2014": 2014,
    "bp_2015": 2015,
    "bp_2016": 2016,
    "bp_2017": 2017,
    "bp_2018": 2018,
    "bp_2019": 2019,
    "bp_2020": 2020,
    "bp_2021": 2021,
    "bp_2022_2023": 2021,
    "bp_unkw": 1900,  # assume if unknown then likely very old
}

logging.info("Convert the building year count columns to integers")
building_age[list(build_dates.keys())] = building_age[build_dates.keys()].astype(int)

logging.info(
    "Multiply build period count columns by the assumed build year to get totals"
)
build_year = building_age[list(build_dates.keys())].mul(build_dates).sum(axis=1)
logging.info("Compute the average age of buildings in each LSOA")
totals = building_age[list(build_dates.keys())].sum(axis=1)
building_age["home_age"] = [2021 - (x / y) for x, y in zip(build_year, totals)]

logging.info("Tidy up columns and merge with main dataset")
building_age = building_age[["ecode", "home_age"]]
building_age.columns = ["LSOA", "home_age"]
df = df.merge(building_age, on="LSOA", how="left")
logging.info(f"Check dataset shape: {df.shape}")

####################### WRITE CLEAN RESULTS TO CSV #############################
logging.info("\n\nClean up columns and write to local csv file")
final_columns = [
    "LA",
    "MSOA",
    "LSOA",
    "temperature",
    "energy_cost",
    "net_income",
    "politically_green",
    "pct_economically_active",
    "home_size",
    "pct_home_occupancy",
    "home_exposed_surfaces",
    "home_age",
    "energy_consumption_per_person",
]
df = df[final_columns]
df.to_csv("compiled_data.csv", index=False)
logging.info("DATA COMPILATION COMPLETE")
