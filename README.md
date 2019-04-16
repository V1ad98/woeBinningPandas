# woeBinningPandas
This code generates a supervised fine and coarse classing of numeric variables and factors with respect to a dichotomous target variable. Its parameters provide flexibility in finding a binning that fits specific data characteristics and practical needs.

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
#### Use Git CMD

>cd YOUR LINK FOLDER

>git clone https://github.com/V1ad98/woeBinningPandas.git

#### File woeBinningPandas is ready to go!
### Set your variable CSV file
> yourvariable = pd.read_csv('Yourfile.csv')
### Set the df variable and specify the column names from your CSV file, which you want to use.
> df = yourvariable[['columnnames1', 'columnnames2','columnnames3']]
### At THE END of the code in the function call woe_binning set the values of the arguments
> binning = woe_binning(df, target_var, pred_var, min_perc_total, min_perc_class, stop_limit, abbrev_fact_levels, event_class)

**df** - Name of data frame with input data.

**target_var** - Name of dichotomous target variable in quotes. Only target variables with two distinct values (0 or 1).

**pred_var** - Name of predictor variables to be binned in quotes. Values can be either numeric or factors.

**min_perc_total** - For numeric variables this parameter defines the number of initial classes before any merging is applied. WOE starts. Increasing the min_perc_total} parameter will avoid sparse bins. Accepted range: 0.0001-0.2; default: 0.05.

**min_perc_ class** - If a column percentage of one of the target classes within a bin is below this limit (e.g. below 0.01=1\%) then the respective bin will be joined with others. In case of numeric variables adjacent predictor classes are merged. 
Setting min_perc_class > 0 may provide more reliable WOE values. Accepted range: 0-0.2; default: 0, i.e. no merging with respect to sparse target classes is applied.

**stop_limit** - Stops WOE based merging of the predictor's classes/levels in case the resulting information value (IV) decreases more than (e.g. 0.05 = 5%) compared to the preceding binning step.
stop_limit=0 will skip any WOE based merging.
Increasing the stop_limit will simplify the binning solution and may avoid overfitting. Accepted range: 0-0.5; default: 0.1.

**abbrev_fact_levels** - Abbreviates the names of new (merged) factor levels via the base abbreviate function in case the specified number of characters is exceeded.

**event_class** - Optional parameter for specifying the class of the target event. This class typically indicates a negative event like a loan default or a disease. Use characters in quotes (e.g. bad).
This class will be represented by negative WOE values then.
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
