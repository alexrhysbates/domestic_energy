# Causal analysis of domestic energy consumption in the UK

An analysis of domestic energy consumption in the UK.

To run the analysis, first install requirements with `pip install -r requirements.txt`.

To compile the dataset from the raw files in the `data/` directory, run `python compile_data.py`

To analyse the data, generate visuals and run the causal analysis, run `python analyse.py` after having compiled the data.

## Approach

* Collate data on financials, energy consumption, demographics, climate and political attitudes by LSOA, MSOA or LA
* Run a Bayesian linear regression to determine isolated impact of each factor on energy consumption per person

## Key results

![image](https://github.com/alexrhysbates/domestic_energy/blob/main/regression_coefficients.png)

* Financial factors are a key driver of consumption - energy price (estimated via relative use of gas vs electric by LSOA) and level of income post housing costs
* Fully-occupied, larger, newer terraced houses and flats are more energy efficient
* Warmer areas use less energy - logical given central heating is responsible for c. 2/3rds of domestic energy consumption
* Areas in England with a higher proportion of Green Party Councillors in England do not necessarily use less energy per person
