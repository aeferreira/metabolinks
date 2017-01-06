from __future__ import print_function
import time
from numpy import nan
import pandas as pd

from six import StringIO, string_types, integer_types

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

    # create a Pandas DataFrame, reading from the list of strings in memory
    mem_string = StringIO('\n'.join(moved_list))
    df = pd.read_table(mem_string)
    return df

def _clean_value(v):
    if not isinstance(v, string_types):
        return v
    try:
        return float(v)
    except:
        pass
    return v.strip('').strip(';')
    
def unfold_MassTRIX_df(df):
    """Unfold a MassTRIX DataFrame.
    
       "Unfolding" means splitting each "peak" line in "compound" lines.
       The # separator is used in the split. The number of Ids found in
       column "KEGG_cid" is used as the number of putative compounds in
       each peak."""
    
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
    
    return pd.DataFrame(unfold_dict, columns=df.columns)


if __name__ == '__main__':

    testfile_name = 'example_data/masses.annotated.reformat.tsv'

    df = read_MassTRIX(testfile_name)
    print("File {} was read".format(testfile_name))

    df.info()

    print('\n+++++++++++++++++++++++++++++++')

    df = unfold_MassTRIX_df(df)

    print('Unfolded dataframe:\n')
    df.info()

    print('---------------------')
    print(df.head(10))
    
    
