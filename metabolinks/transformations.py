"""All transformations should accept a pandas DataFrame object.
   Most should return a new pandas DataFrame."""
from functools import partial

import numpy as np
import pandas as pd

# ---------- imputation of missing values -------

def fillna_zero(df):
    return df.fillna(0.0)

def _fillna_value(df, value):
    return df.fillna(value)

# def fillna_value(value=0.0):
#     return partial(_fillna_value, value=value)

def fillna_value(df, value=0.0):
    return df.fillna(value)
