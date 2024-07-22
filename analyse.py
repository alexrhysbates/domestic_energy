"""
Script to analyse energy consumption data.

Compiled data can be found in compiled_data.csv. The script works by first reading in this data.

Running `pip install -r requirements.txt` installs all necessary dependencies for this project.

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
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
RANDOM_SEED = 1999

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
plt.title("Domestic energy consumption per person across the UK")
logging.info("Saving plot to a local .png")
plt.savefig('energy_consumption_per_person_by_uk_local_authority.png')
plt.close()


############# EXPLORATORY ANALYSIS ############################
logging.info("\n\nExploratory analysis on dataset")
logging.info("Plot a correlation heatmap of all the features in the dataset")
logging.info("Compute correlations of all the feature columns and plot using a seaborn heatmap")
corr_data = df.drop(columns=['LA', 'MSOA', 'LSOA'], axis=1).corr()
corr_heatmap = sns.heatmap(corr_data, cmap="crest", annot=True)
corr_heatmap.set_title("Correlations of features related to domestic energy consumption") 
corr_heatmap.set(xlabel="", ylabel="")
corr_heatmap.xaxis.tick_bottom()
plt.xticks(rotation=90)
logging.info("Saving plot to a local .png")
plt.savefig("correlations_of_features.png", bbox_inches="tight")
plt.close()

logging.info("Plotting histogram of each variable, saving to local .png")
df.hist(bins=20, figsize=(10,10), color="purple")
plt.suptitle("Distribution of features related to domestic energy consumption")
plt.savefig("histogram_of_features.png")
plt.close()

############# CAUSAL ANALYSIS #################################
logging.info("\n\nRunning a causal analysis of the data")  
model_df = df[['LA', 'politically_green','temperature','energy_cost',
            'net_income','pct_economically_active', 'home_size',
            'pct_home_occupancy', 'home_exposed_surfaces', 'home_age',
            'energy_consumption_per_person']].copy()
logging.info("Clean the data, drop any rows with nulls assume data is missing at random due to merging on LSOA. Fix dtype issues")
model_df.dropna(how='any',axis=0, inplace=True)
model_df["net_income"] = model_df["net_income"].astype(float)
logging.info("Convert Local authority to a categorical code column that we can index into, if needed")
model_df['LA'] = model_df['LA'].astype('category').cat.codes

logging.info("Normalising continuous variables to z-scores with mean 0 and variance 1")
scaler = StandardScaler()
scaler.fit(model_df.iloc[:,2:]) # don't normalise non-continuous "LA" or "politically_green" features
model_df.iloc[:,2:] = scaler.transform(model_df.iloc[:,2:])
logging.info(f"Model data prepared, with shape {model_df.shape}. \nAnd features: {model_df.columns[1:-1]}. \nAnd target variable: {model_df.columns[-1]}")

logging.info("Defining the probablistic model in pymc")
with pm.Model() as model:
    
    logging.info("Set uninformative priors for model parameters")
    a = pm.Normal('a', 0, 1) # a = intercept
    b_temperature = pm.Normal('b_temperature', 0, 1) # b_* parameters are slope parameters in our linear model
    b_energy_cost = pm.Normal('b_energy_cost', 0, 1)
    b_net_income = pm.Normal('b_net_income', 0, 1)
    b_pct_economically_active = pm.Normal('b_pct_economically_active', 0, 1)
    b_politically_green = pm.Normal('b_politically_green', 0, 1)
    b_home_size = pm.Normal('b_home_size', 0, 1)
    b_pct_home_occupancy = pm.Normal('b_pct_home_occupancy', 0, 1)
    b_home_exposed_surfaces = pm.Normal('b_home_exposed_surfaces', 0, 1)
    b_home_age = pm.Normal('b_home_age', 0, 1)
    sigma = pm.Exponential('sigma', 1)
    
    logging.info("Define mean energy consumption per person with a linear model of the features")
    mu = pm.Deterministic('mu', a + 
                          b_politically_green * model_df.politically_green.values + 
                          b_temperature * model_df.temperature.values +
                          b_energy_cost * model_df.energy_cost.values +
                          b_net_income * model_df.net_income.values +
                          b_pct_economically_active * model_df.pct_economically_active.values +
                          b_home_size * model_df.home_size.values +
                          b_pct_home_occupancy * model_df.pct_home_occupancy.values +
                          b_home_exposed_surfaces * model_df.home_exposed_surfaces.values +
                          b_home_age * model_df.home_age.values
                          ) 
    
    logging.info("Our likelihood is the based on an assumed normal distribution of energy consumption \nwith its mean defined by a linear model") 
    likelihood = pm.Normal('likelihood', mu = mu, sigma = sigma, observed = model_df.energy_consumption_per_person.values)
    
logging.info("The posterior is analytically intractable, so we approximate by sampling using MCMC")
n_samples = 2_000
logging.info(f"Take {n_samples} samples from the posterior distribution for each model parameter. \nNote that this may take a few minutes to run")  
with model:
    trace = pm.sample(n_samples, tune=1000, random_seed=RANDOM_SEED) # take 2,000 samples from the posterior
    
logging.info("Plot the distribution of the regression coefficents and save to a local .png") 
with model:
    az.plot_forest(trace,
                   kind='ridgeplot',
                   filter_vars="regex",
                   var_names=["^b"], # only include b_* parameters
                   hdi_prob=0.99, # capture the highest density 99% of each param distribution
                   textsize = 8.0,
                   figsize=(5, 3),
                   colors="purple",
                   combined=True)
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.title("What is the isolated impact of each driver on energy consumption?")
plt.savefig("regression_coefficients.png")
plt.close()

logging.info("Look at the mean predictions for energy consumption from the model, \nand compute model goodness of fit with R-squared")
with model:
    predictions = pm.sample_posterior_predictive(trace, model, random_seed=RANDOM_SEED) # sample from posterior for energy consumption
y_pred = np.mean(predictions["posterior_predictive"].likelihood[0], axis=0) # compute the mean for each LSOA
y_true = model_df.energy_consumption_per_person.values
score = r2_score(y_true, y_pred) # calculate the R2 score for how well the model explains the data
logging.info(f"R-squared for model goodness of fit = {round(score,2)}")
logging.info("Analysis complete.")