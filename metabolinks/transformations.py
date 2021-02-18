"""All transformations should accept a pandas DataFrame object.

   Most should return a new pandas DataFrame.
   The input data matrix should follow the convention that the
   instances are in rows and features are in columns.
"""
from functools import partial

import numpy as np
import pandas as pd

import metabolinks as mtls
from metabolinks.utils import _is_string

# ---------- imputation of missing values -------

def fillna_zero(df):
    """Set NaN to zero."""
    return df.fillna(0.0)

def fillna_value(df, value=0.0):
    """Set NaN to value."""
    return df.fillna(value)

def fillna_frac_min(df, fraction=0.5):
    """Set NaN to a fraction of the minimum value in whole DataFrame."""

    minimum = df.min().min()
    # print(minimum)
    minimum = minimum * fraction
    return df.fillna(minimum)

# ---------- filters for reproducibility

def keep_atleast(df, min_samples=1):
    """Keep only features wich occur at least in min_samples.

       If 0 < min_samples < 1, this should be interpreted as a fraction."""

    counts = df.count(axis=1)
    # a float in (0,1) means a fraction
    if 0.0 < min_samples < 1.0:
        n = len(df.columns)
        min_samples = min_samples * float(n)

    return df[counts >= min_samples]

def keep_atleast_inlabels(df, min_samples=1):
    """Keep only features wich occur at least in min_samples in each label.

       If 0 < min_samples < 1, this should be interpreted as a fraction.
       Features may not be removed: they are marked as NaN in each label
       with less than min_samples, but can be kept because of their presence
       in other labels. Nevertheless, a final removal of 'all Nan' features is performed."""

    # print('****** df *********')
    # print(df)
    # print('*******************')
    for label in df.cdl.unique_labels:
        # get a copy of label data
        df_label = df.cdl.subset(label=label)
        old_index = df_label.index.copy()
        # print('----- Label {} -------'.format(label))
        # print('------------------------')
        # print(df_label)

        counts = df_label.count(axis=1)
        # a float in (0,1) means a fraction
        if 0.0 < min_samples < 1.0:
            n = len(df_label.columns)
            min_samples = min_samples * float(n)
        df_label = df_label[counts >= min_samples].reindex(old_index, method=None)
        # print('------ after removal ---')
        # print(df_label)
        bool_loc = df.cdl.subset_where(label=label)
        df = df.mask(bool_loc, df_label)
        # print('+++++++ current df +++++++')
        # print(df)
        # print('++++++++++++++++++++++++++')
    # print('****** final df *********')
    # print(df.dropna(how='all'))
    # print('*************************')
    return df.dropna(how='all')

# ---------- normalizations

def normalize_ref_feature(df, feature, remove=True):
    """Normalize dataset by a reference feature (an exact row label).

       df: a Pandas DataFrame.
       feature: row label.
       remove: bool; True to remove reference feature from data after normalization.

       Returns: DataFrame.
       """

    # find position of feature
    new_index, indexer = df.index.sort_values(return_indexer=True)
    pos = new_index.get_loc(feature, method='pad')
    pos = indexer[pos]
    feature_row = df.iloc[pos, :]
    df = df / feature_row
    if remove:
        df = df.drop(index=[df.index[pos]])
    return df

def remove_feature(df, feature):
    """Remove a reference feature (an exact row label).

       df: a Pandas DataFrame.
       feature: row label.

       Returns: DataFrame.
       """

    # find position of feature
    new_index, indexer = df.index.sort_values(return_indexer=True)
    pos = new_index.get_loc(feature, method='pad')
    pos = indexer[pos]
    df = df.drop(index=[df.index[pos]])
    return df

def normalize_sum (df):
    """Normalization of a dataset by the total value per columns."""

    return df/df.sum(axis=0)

# Needs double-checking
def normalize_PQN(df, ref_sample='mean'):
    """Normalization of a dataset by the Probabilistic Quotient Normalization method.

       df: Pandas DataFrame.
       ref_sample: reference sample to use in PQ Normalization, types accepted: "mean" (default, reference sample will be the intensity
    mean of all samples for each feature - useful for when there are a lot of imputed missing values), "median" (reference sample will
    be the intensity median of all samples for each feature - useful for when there aren't a lot of imputed missing values), column name
    of the sample - ('label','sample') if data is labeled (reference sample will be the sample with said column name in the dataset)  -
    or list with the intensities of all peaks that will directly be the reference sample (pandas Series not accepted - list(Series) is
    accepted).

       Returns: Pandas DataFrame; normalized spectra.
    """
    #Total Int normalization first - MetaboAnalyst doesn't do it but paper recommends it?
    #"Building" the reference sample based on the input given
    if ref_sample == 'mean': #Mean spectre of all samples
        ref_sample2 = df.T / df.mean(axis = 1)
    elif ref_sample == 'median': #Median spectre of all samples
        ref_sample2 = df.T/df.median(axis = 1)
    elif ref_sample in df.columns: #Column name of a specifiec sample of the spectra. ('Label','Sample') if data is labeled
        ref_sample2 = df.T/df.loc[:,ref_sample]
    else: #Actual sample given
        ref_sample2 = df.T/ref_sample
    #Normalization Factor and Normalization
    Norm_fact = ref_sample2.median(axis=1)
    return df / Norm_fact


def normalize_quantile(df, ref_type='mean'):
    """Normalization of a dataset by the Quantile Normalization method.

       Missing Values are temporarily replaced with 0 (and count as 0) until normalization is done. Quantile Normalization is more
    useful with no/low number of missing values.

       Spectra: AlignedSpectra object (from metabolinks).
       ref_type: str (default: 'mean'); reference sample to use in Quantile Normalization, types accepted: 'mean' (default,
    reference sample will be the means of the intensities of each rank), 'median' (reference sample will be the medians of the
    intensities for each rank).

       Returns: AlignedSpectra object (from metabolinks); normalized spectra.
    """
    #Setting up the temporary dataset with missing values replaced by zero and dataframes for the results
    norm = df.copy().replace({np.nan:0})
    ref_spectra = df.copy()
    ranks = df.copy()

    for i in range(len(norm.columns)):
        #Determining the ranks of each feature in the same column (same sample) in the dataset
        ref_spectra.iloc[:,i] = norm.iloc[:,i].sort_values().values
        ranks.iloc[:,i] = norm.iloc[:,i].rank(na_option='top')

    #Determining the reference sample for normalization based on the ref_type chosen (applied on the columns).
    if ref_type == 'mean':
        ref_sample = ref_spectra.mean(axis=1).values
    elif ref_type == 'median':
        ref_sample = ref_spectra.median(axis=1).values
    else:
        raise ValueError('Type not recognized. Available ref_type: "mean", "median".')

    #Replacing the values in the dataset for the reference sample values based on the ranks calculated  earlier for each entry
    for i in range(len(ranks)):
        for j in range(len(ranks.columns)):
            if ranks.iloc[i,j] == round(ranks.iloc[i,j]):
                norm.iloc[i,j] = ref_sample[int(ranks.iloc[i,j])-1]
            else: #in case the rank isn't an integer and ends in .5 (happens when a pair number of samples have the same
                  #value in the same column - after ordering from lowest to highest values by row).
                norm.iloc[i,j] = np.mean((ref_sample[int(ranks.iloc[i,j]-1.5)], ref_sample[int(ranks.iloc[i,j]-0.5)]))

    #Replacing 0's by missing values and creating the AlignedSpectra object for the output
    return norm.replace({0:np.nan})


# ---------- log transformations and scalings

def glog(df, lamb=None):
    """Performs Generalized Logarithmic Transformation on a Spectra (same as MetaboAnalyst's transformation).

       df: Pandas DataFrame.
       lamb: scalar, optional (default: minimum value in the data divided by 10); transformation parameter lambda.

       Returns: DataFrame transformed as log2(y + (y**2 + lamb**2)**0.5).
       """
    # Default lambda
    if lamb is None:
        lamb = min(df.min())/10.0
    # Apply the transformation
    y = df.values
    y = np.log2((y + (y**2 + lamb**2)**0.5)/2)
    return pd.DataFrame(y, index=df.index, columns=df.columns)


def pareto_scale(df):
    """Performs Pareto Scaling on a DataFrame."""

    means = df.mean(axis=1)
    stds = df.std(axis=1) ** 0.5
    df2 = df.sub(means, axis=0).div(stds, axis=0)
    return df2


def mean_center(df):
    """Performs Mean Centering.

       df: Pandas DataFrame. It can include missing values.

       Returns: DataFrame; Mean Centered Spectra."""
    return df.sub(df.mean(axis=1), axis=0)


def auto_scale(df):
    """Performs Autoscaling on column-organized data.

       Returns: Pandas DataFrame; Auto Scaled Spectra.

       This is x -> (x - mean(x)) / std(x) per feature"""

    # TODO: verify if the name of this transformation is "Standard scaling"
    # TODO: most likely it is known by many names (scikit-learn has a SatndardScaler transformer)
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    df2 = df.sub(means, axis=0).div(stds, axis=0)
    return df2

def range_scale(df):
    """Performs Range Scaling.

       Returns: Pandas DataFrame."""

    scaled_aligned = df.copy()
    ranges = df.max(axis=1) - df.min(axis=1) # Defining range for every feature
    # Applying Range scaling to each feature
    for j in range(0, len(scaled_aligned)):
        if ranges.iloc[j] == 0: # No difference between max and min values
            scaled_aligned.iloc[j, :] = df.iloc[j, ]
        else:
            scaled_aligned.iloc[j, :] = (df.iloc[j, ] - df.iloc[j, ].mean()) / ranges.iloc[j]

    return scaled_aligned


def vast_scale(df):
    """Performs Vast Scaling.

       Returns: Pandas DataFrame; Vast Scaled Spectra."""

    # scaled_aligned = df.copy()
    std = df.std(axis=1)
    mean = df.mean(axis=1)
    # Applying Vast Scaling to each feature
    scaled_aligned = (((df.T - mean)/std)/(mean/std)).T

    # Return scaled spectra
    return scaled_aligned


def level_scale(df, average=True):
    """Performs Level Scaling on a DataFrame. (See van den Berg et al., 2006).

    average: bool (Default - True); if True mean-centered data is divided by the mean spectra, if False it is divided by the median
    spectra.

    Returns: Pandas DataFrame; Level Scaled Spectra."""

    mean = df.mean(axis=1)
    # Applying Level Scaling to each feature
    if average == True:
        scaled_aligned = ((df.T - mean)/mean).T
    elif average == False:
        scaled_aligned = ((df.T - mean)/df.median(axis=1)).T
    else:
        raise ValueError ('Average is a boolean argument. Only True or False accepted.')

    # Return scaled spectra
    return scaled_aligned


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
    from metabolinks.datasets import demo_dataset
    # read sample data set demo2
    print('\nLoad demo2: data with labels ------------\n')
    data = demo_dataset('demo2').data.transpose()
    print(data)
    print('-- info --------------')
    print(data.cdl.info())
    print('-- global info---------')
    print(data.cdl.info(all_data=True))
    print('-----------------------')

    print('\n--- fillna_zero ----------')
    new_data = fillna_zero(data)
    print(new_data)
    print('--- fillna_value  10 ----------')
    new_data = fillna_value(data, value=10)
    print(new_data)
    print('--- fillna_frac_min default fraction=0.5 minimum ----------')
    new_data = fillna_frac_min(data)
    print(new_data)

    print('\n--- keep_atleast min_samples=3 ----------')
    new_data = keep_atleast(data, min_samples=3)
    print(new_data)
    print('--- keep_atleast min_samples=5/6 ----------')
    new_data = keep_atleast(data, min_samples=5/6)
    print(new_data)
    print('\n--- keep_atleast_inlabels min_samples=2 ----------')
    new_data = keep_atleast_inlabels(data, min_samples=2)
    print(new_data)

    print('\nNormalization by reference feature ------------\n')
    print('---- original -------------------')
    print(data)
    print('------after normalizing by 97.59001 -----------------')
    new_data = normalize_ref_feature(data, 97.59001)
    print(new_data)

    # read sample data set
    print('\npareto scaling ------------\n')
    print('---- original -------------------')
    print(data)
    print('------after Pareto scaling -----------------')
    new_data = pareto_scale(data)
    print(new_data)
