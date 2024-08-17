# Causal analysis of domestic energy consumption in the UK

An analysis of domestic energy consumption in the UK.

To run the analysis, first install requirements with `pip install -r requirements.txt`.

To compile the dataset from the raw files in the `data/` directory, run `python compile_data.py`

To analyse the data, generate visuals and run the causal analysis, run `python analyse.py` after having compiled the data.

## Approach

* Collate data on financials, energy consumption, demographics and political attitudes by LSOA, MSOA or LA
* Run a Bayesian linear regression to determine isolated impact of each factor

## Key results

![image](https://github.com/alexrhysbates/domestic_energy/blob/main/regression_coefficients.png)

