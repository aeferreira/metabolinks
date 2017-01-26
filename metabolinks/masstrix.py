from __future__ import print_function, absolute_import
from six import StringIO, string_types, integer_types

import time

from numpy import nan
import pandas as pd

from metabolinks.taxonomy import insert_taxonomy

"""Functions related to the analysis of MassTRIX results files."""

# ----------------------------------------
# MassTRIXResults
# ----------------------------------------

class MassTRIXResults(pd.DataFrame):
    """A subclass of Pandas DataFrame representing MassTRIX identifications.
    
       Created to facilitate method chaining, and sparing the name space.
       Instances are created mainly by reading from file functions.
    """
    @property
    def _constructor(self):
        return MassTRIXResults    
    
##     def __init__(self, other):
##         # Copy attributes only if other is of good type
##         if isinstance(other, pd.DataFrame):
##             self.__dict__  = other.__dict__.copy()

    def cleanup_cols(self, **kwargs):
        return cleanup_cols(self, **kwargs)
    
    def unfold(self, **kwargs):
        return unfold(self, **kwargs)
    
    def insert_taxonomy(self, *args, **kwargs):
        return insert_taxonomy(self, *args, **kwargs)

# ----------------------------------------
# I/O functions
# ----------------------------------------

def read_MassTRIX(fname):
    """Reads a MassTRIX file into a Pandas DataFrame object.
       
       On the process, the last line is moved to the beginning and
       is read as the header."""
    
    # store lines in a list
    with open(fname) as f:
        lines = [line.strip() for line in f]
    
    # move the last line to the beginning
    moved_list = [lines[-1]]
    moved_list.extend(lines[:-1]) # last line is not included

    # create a MassTRIXResults, which is also a Pandas DataFrame,
    # reading from the list of strings in memory
    
    mem_string = StringIO('\n'.join(moved_list))
    
    df = pd.read_table(mem_string)
    df = MassTRIXResults(df)
    return df

# ----------------------------------------
# Unfold functions
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
    if not isinstance(v, string_types):
        return v
    try:
        return float(v)
    except:
        pass
    return v.strip('').strip(';')


def unfold(df):
    """Unfold a MassTRIX DataFrame.
    
       "Unfolding" means splitting each "peak" line into several
       "compound" lines. # is the split separator.
       
       The number of Ids found in
       column "KEGG_cid" is used as the number of putative compounds in
       each peak.
    """
    
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
    
    for indx, row in df.iterrows():
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
    
    return MassTRIXResults(unfold_dict, columns=df.columns)

# ----------------------------------------
# Clean up functions
# ----------------------------------------

def cleanup_cols(df, isotopes=True, uniqueID=True, columns=None):
    """Removes the 'uniqueID' and the 'isotope presence' columns."""
    col_names = []
    if uniqueID:
        col_names.append('uniqueID')
    if isotopes:
        iso_names = ('C13','O18','N15', 'S34', 'Mg25', 'Mg26', 'Fe54',
                     'Fe57', 'Ca44', 'Cl37', 'K41')
        col_names.extend(iso_names)
    if columns is not None:
        col_names.extend(columns)
    #df.drop(col_names, axis=1, inplace=True)
    return df.drop(col_names, axis=1)
    

# ----------------------------------------
# Testing code
# ----------------------------------------

if __name__ == '__main__':

    testfile_name = '../example_data/masses.annotated.reformat.tsv'

    results = read_MassTRIX(testfile_name).cleanup_cols()
    
    print("File {} was read\n".format(testfile_name))
    
    results.info()

    print('\n+++++++++++++++++++++++++++++++')

    results = results.unfold()

    print('Unfolded dataframe:\n')
    print('is of type', type(results))
    results.info()

    print('---------------------')
    print(results.head(10))
    
    
