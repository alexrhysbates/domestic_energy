# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3.11
#     language: python
#     name: py311
# ---

# %% [markdown]
# # Experimenting with possible approaches

# %% jupyter={"outputs_hidden": true}
# !pip install -r ../requirements.txt

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd
import pymc as pm
import geopy

# %% [markdown]
# ## 1. Data compilation

# %%
# assumptions
# source - https://www.icaew.com/insights/viewpoints-on-the-news/2022/sept-2022/chart-of-the-week-energy-price-cap-update
GAS_PRICE_PER_KWH = 3.3
ELECTRIC_PRICE_PER_KWH = 19.0 

# %%
# look at the headline dataset of consumption by LSOA
main_data = pd.read_csv("../data/raw/LSOA Energy Consumption Data.csv")

# %%
main_data.shape

# %%
main_data['Lower Layer Super Output Area (LSOA) Code'].nunique()

# %%
main_data.columns

# %%
# look at household size data
household_size = pd.read_csv("../data/raw/RM202-Household-Size-By-Number-Of-Rooms-2021-lsoa-ONS.csv")

# %%
household_size["Number of rooms (Valuation Office Agency) (6 categories) Code"].unique()

# %%
household_size[household_size["Number of rooms (Valuation Office Agency) (6 categories) Code"] == 4]["Household size (5 categories) Code"].unique()

# %%
building_age = pd.read_csv("../data/raw/CTSOP_4_1_2021.csv")

# %%
# Library to work with netCDF files
from netCDF4 import Dataset

file_name = "../data/raw/tas_hadukgrid_uk_60km_ann_202101-202112.nc"
file_id = Dataset(file_name)

latitude = file_id.variables["latitude"][:,:]
longitude = file_id.variables["longitude"][:,:]
temps = file_id.variables["tas"][:,:]

lats = [np.mean(x) for x in latitude]
longs = [np.mean(x) for x in longitude] 
ts = [np.mean(x) for x in temps[0]]
temp_data = pd.DataFrame({"latitude": lats,
                          "longitude": longs,
                          "temperature": ts}
                        )

temp_data = temp_data[temp_data.temperature > 0]

# %% [markdown]
# ### Combining and generating features

# %%
# feature generation
main_data["pct_electric"] = main_data['Electricity Consumption (kWh)'] / main_data['Total Energy Consumption (kWh)']
main_data["coords"] = [(lat, long) for lat, long in zip(main_data.Latitude, main_data.Longitude)]

# %%
df = main_data[['Local Authority Name', 'Local Authority Code', 'MSOA Name',
       'Middle Layer Super Output Area (MSOA) Code', 'LSOA Name',
       'Lower Layer Super Output Area (LSOA) Code', 'coords',
       'pct_electric', 'Average Energy Consumption per Person (kWh)']]

# %%
df.columns = ['LA_name', 'LA', 'MSOA_ame',
       'MSOA', 'LSOA_name',
       'LSOA', 'coords',
       'pct_electric', 'energy_consumption_per_person']

# %%
# add temperature data
coords =  [(lat, long) for lat, long in zip(temp_data.latitude, temp_data.longitude)]
temp_dict = {co:t for co,t in zip(coords, temp_data.temperature)}

def find_closest_temp_measurement(this_point):
    return temp_dict[min(temp_dict.keys(), key=lambda x: geopy.distance.geodesic(this_point, x))]

df["temperature"] = [find_closest_temp_measurement(x) for x in df.coords]

# %%
df.temperaure.value_counts()

# %% [markdown]
# ## 2. Analysis

# %%

# %%

# %% [markdown]
# ## 3. Modelling

# %%
