"""All transformations should accept a pandas DataFrame object.
   Most should return a new pandas DataFrame."""
from functools import partial

import numpy as np
import pandas as pd

from metabolinks import msaccessor
from metabolinks.utils import _is_string

# ---------- imputation of missing values -------

def fillna_zero(df):
    return df.fillna(0.0)

def fillna_value(df, value=0.0):
    return df.fillna(value)

def fillna_frac_min(df, fraction=0.5):
    minimum = df.min().min()
    # print(minimum)
    minimum = minimum * fraction
    return df.fillna(minimum)

# ---------- filters for reproducibility


# ----------------------------------------
# MassTRIX related functions
#
# these relate to the two MassTRIX formats:
#
# - compact: one peak per line, several putative compounds
#   are separated by #
#
# - unfolded: one putative compound per line
# 
# ----------------------------------------

def _clean_value(v):
    if not _is_string(v):
        return v
    try:
        return float(v)
    except:
        pass
    return v.strip('').strip(';')


def unfold_MassTRIX(df):
    """Unfold a MassTRIX DataFrame.
    
       "Unfolding" means splitting each "peak" line into several
       "compound" lines. # is the split separator.
       
       The number of Ids found in
       column "KEGG_cid" is used as the number of putative compounds in
       each peak.
    """
    nan = np.nan
    # split by #, if possible, using str accessor of Pandas Series.
    for c in df.columns:
        try:
            df[c] = df[c].str.split('#')
        except AttributeError:
            pass
    
    # initialize a dict of empty lists, with column names as keys
    unfold_dict = {c:[] for c in df.columns}
    
    # unfold each row of the data frame.
    # if a column is of type list, extend the appropriate lists
    # otherwise repeat elements using row['KEGG_cid'] as the number of repeats
    
    for _, row in df.iterrows():
        n = len(row['KEGG_cid'])
    
        for label, v in row.iteritems():
            if isinstance(v, list):
                if len(v) < n:
                    v.extend([nan]*(n-len(v)))
                unfold_dict[label].extend(v)
            else:
                unfold_dict[label].extend([v] * n)
    
    # clean values (try to convert to float and remove leading ;)
    for k in unfold_dict:
        unfold_dict[k] = [_clean_value(v) for v in unfold_dict[k]]
    
    return pd.DataFrame(unfold_dict, columns=df.columns)

###################################################################

if __name__ == "__main__":
    import six
    from metabolinks import datasets
    from metabolinks import dataio
    # read sample data set
    print('\nReading sample data with labels (as io stream) ------------\n')
    data = dataset = dataio.read_data_csv(six.StringIO(datasets.demo_data2()), has_labels=True)
    print(dataset)
    print('-- info --------------')
    print(dataset.ms.info())
    print('-- global info---------')
    print(dataset.ms.info(all_data=True))
    print('-----------------------')

    print('--- fillna_zero ----------')
    new_data = fillna_zero(dataset)
    print(new_data)
    print('--- fillna_value  10 ----------')
    new_data = fillna_value(dataset, value=10)
    print(new_data)
    print('--- fillna_value ----------')
    new_data = fillna_frac_min(dataset)
    print(new_data)


