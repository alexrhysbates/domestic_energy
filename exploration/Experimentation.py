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

# %% [markdown]
# ## 1. Data compilation

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
main_data["Shape_Area"].values

# %%
# look at household size data
household_size = pd.read_csv("../data/raw/RM202-Household-Size-By-Number-Of-Rooms-2021-lsoa-ONS.csv")

# %%
household_size["Number of rooms (Valuation Office Agency) (6 categories) Code"].unique()

# %%
household_size[household_size["Number of rooms (Valuation Office Agency) (6 categories) Code"] == 4]["Household size (5 categories) Code"].unique()

# %%
building_age = pd.read_csv("../data/raw/CTSOP_4_1_2021.csv")

# %% [markdown]
# ## 2. Analysis

# %%

# %% [markdown]
# ## 3. Modelling

# %%
