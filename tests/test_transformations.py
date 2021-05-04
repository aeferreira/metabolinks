import pytest
import numpy as np
import pandas as pd
from io import StringIO
#from metabolinks import parse_data
#from metabolinks.dataio import gen_df, read_data_csv
from metabolinks.datasets import demo_dataset
from metabolinks import transformations as trans

# m/z            97.58868   97.59001   97.59185  ...   98.57790   98.57899   99.28772
# label sample                                   ...
# l1    s38     1073218.0   637830.0   460092.0  ...  3891025.0        NaN  2038979.0
# l2    s39     1049440.0   534900.0   486631.0  ...  3990442.0  1877649.0        NaN
# l1    s40     1058971.0   582966.0        NaN  ...  3888258.0  1864650.0        NaN
# l2    s32     2351567.0  1440216.0  1137139.0  ...  2133404.0  1573559.0        NaN   
#       s33     1909877.0  1124346.0   926038.0  ...        NaN        NaN  3476845.0   
#       s34     2197036.0  1421899.0  1176756.0  ...  3643682.0  1829208.0        NaN

# [6 rows x 19 columns]


def test_fillna_zero():
    """test construction from demo1 data set (CSV with no labels)."""
    demo2 = demo_dataset('demo2')
    data = demo2.data
    #y = demo2.target

    new_data = trans.fillna_zero(data)
    
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19)
    assert new_data.iloc[2, 2] == 0.0

def test_fillna_value():
    """test construction from demo1 data set (CSV with no labels)."""
    demo2 = demo_dataset('demo2')
    data = demo2.data
    #y = demo2.target

    new_data = trans.fillna_value(data, value=10.0)
    
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19)
    assert new_data.iloc[2, 2] == 10.0

def test_fillna_frac_min():
    """test construction from demo1 data set (CSV with no labels)."""
    demo2 = demo_dataset('demo2')
    data = demo2.data

    globalmin = 0.5 * data.min().min()
    #y = demo2.target

    new_data = trans.fillna_frac_min(data, fraction=0.5)
    
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19)
    assert new_data.iloc[2, 2] == globalmin