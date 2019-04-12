# woeBinningPandas
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
#### Set your variable CSV file
> yourvariable = pd.read_csv('Yourfile.csv')
#### Set the df variable and specify the column names from your CSV file, which you want to use.
> df = yourvariable[['columnnames1', 'columnnames2','columnnames3']]
#### At the end of the code in the function call woe_binning set the values of the arguments
> woe_binning (df, target_var, pred_var, min_perc_total, min_perc_class, stop_limit, abbrev_fact_levels, event_class)

> target_var = column_names_with_0_or_1

> pred_var = column_names_with_numbers_or_factors

# Using with PIP package
#### Download PIP package woeBinningPandas
> pip install woeBinningPandas
#### Add all used libraries
> import pandas as pd

> import numpy as np

> import math

> import warnings

> import copy

> import woeBinningPandas
#### Set variables and call a function
> yourvariable = pd.read_csv('Yourfile.csv')

> df = yourvariable[['columnnames1', 'columnnames2','columnnames3']]

> print(woeBinningPandas.woe_binning (df, target_var, pred_var, min_perc_total, min_perc_class, stop_limit, abbrev_fact_levels, event_class))
