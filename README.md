# woeBinningPandas
This program allows simplify the work in credit scoring, by finding Information Value (IV) and Weight of evidence (WOE) from a CSV file or data set.
The basis of this code was taken woeBinning code (https://github.com/cran/woeBinning/blob/master/R/woe.binning.R) and changed from R to Python.
# Used programs versions
Spyder (Python 3.7)

Pandas 0.23.4
# Used Python libraries
import pandas as pd

import numpy as np

import math

import warnings

import copy
# Using
### Cloning a repository from GitHub
Use Git CMD

>cd YOUR LINK FOLDER

>git clone https://github.com/V1ad98/woeBinningPandas.git

File woeBinningPandas is ready to go!
### Set your variable CSV file
> yourvariable = pd.read_csv('Yourfile.csv')
### Set the df variable and specify the column names from your CSV file, which you want to use.
> df = yourvariable[['columnnames1', 'columnnames2','columnnames3']]
### At THE END of the code in the function call woe_binning set the values of the arguments
> binning = woe_binning(df, target_var, pred_var, min_perc_total, min_perc_class, stop_limit, abbrev_fact_levels, event_class)

target_var - df dataframe column in which the values are only 0 or 1

pred_var - df dataframe column in which values can be either numeric or factors

# Using with PIP package
### Download PIP package woeBinningPandas
> pip install woeBinningPandas
### Add use package
> import woeBinningPandas
### Set variables and call a function
> yourvariable = woeBinningPandas.pd.read_csv('Yourfile.csv')

> df = yourvariable[['columnnames1', 'columnnames2','columnnames3']]
### Pass your values to functions
> binning = woeBinningPandas.woe_binning (df, target_var, pred_var, min_perc_total, min_perc_class, stop_limit, abbrev_fact_levels, event_class)
# Examples
> import woeBinningPandas

> germancredit = woeBinningPandas.pd.read_csv('GermanCredit.csv')

> df = germancredit[['credit_risk', 'amount','duration']]

> binning = woeBinningPandas.woe_binning(df, 'credit_risk', 'duration', 0.05, 0, 0.1, 50, 'bad')
