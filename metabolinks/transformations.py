"""All transformations should accept a pandas DataFrame object.
   Most should return a new pandas DataFrame."""
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
    for label in df.ms.unique_labels:
        # get a copy of label data
        df_label = df.ms.subset(label=label)
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
        bool_loc = df.ms.subset_where(label=label)
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

# ---------- log transformations and scaling

def glog(df, lamb=None):
    """Performs Generalized Logarithmic Transformation on a Spectra (same as MetaboAnalyst's transformation).

       Spectra: AlignedSpectra object (from metabolinks).
       lamb: scalar, optional (default: True);  transformation parameter, if True lamb = minimum value in the data divided by 10.

       Returns: AlignedSpectra object (from metabolinks); transformed spectra by a factor of log(y + (y**2 + lamb**2)**0.5).
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

    # scaled_aligned = df.data.copy()
    # for j in range(0, len(scaled_aligned)):
    #     std = df.data.iloc[j, ].std()
    #     sqstd = std**(0.5)
    #     values = df.data.iloc[j, ]
    #     # Apply Pareto Scaling to each value
    #     values = (values - values.mean())/sqstd
    #     # Replace not null values by the scaled values
    #     if len(values) == df.sample_count:
    #         scaled_aligned.iloc[j, :] = values
    #     else:
    #         a = 0
    #         for i in range(0, len(df.sample_count)):
    #             if scaled_aligned.notnull().iloc[j, i]:
    #                 scaled_aligned.iloc[j, i] = values.iloc[a, 0]
    #                 a = a + 1

    # # Return scaled spectra
    # return AlignedSpectra(scaled_aligned, sample_names=Spectra.sample_names, labels=Spectra.labels)

# def spectra_proc(Spectra, minsample=0, Feat_mass=False, remove=True, lamb= 'False', Pareto = True):
#     """Performs any combination of Missing Value Imputation, Normalization by a reference feature, Generalized Logarithmic 
#     Transformation and Pareto Scaling of the dataset.

#        Spectra: Aligned Spectra object (from metabolinks). It can include missing values.
#        minsample: scalar, optional; number between 0 and 1, minsample*100 represents the minimum % of samples where the feature must 
#     be present in order to not be removed.
#        Feat_mass: scalar (default: False); m/z of the reference feature to normalize the sample. False - Normalization is not performed.
#        remove: bool (deafult: True); True to remove reference feature from data after normalization.
#        lamb: scalar (default - 'False');  transformation parameter, if 'False', glog transformation is not performed.
#        Pareto: bool (default - True); if True performs Pareto Scaling.

#        Returns: Processed Aligned Spectra object (from metabolinks).
#     """
#     if minsample != 100: #Missing Value Imputation
#         Spectra = NaN_Imputation(Spectra, minsample)

#     if Feat_mass != False: #Normalization by a reference feature
#         Spectra = Norm_Feat(Spectra, Feat_mass, remove = remove)

#     if lamb != 'False': #glog transformation
#         Spectra = glog(Spectra, lamb)

#     if Pareto != False: #Pareto Scaling
#         Spectra = ParetoScal(Spectra)
#     return Spectra


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

    print('\n--- fillna_zero ----------')
    new_data = fillna_zero(dataset)
    print(new_data)
    print('--- fillna_value  10 ----------')
    new_data = fillna_value(dataset, value=10)
    print(new_data)
    print('--- fillna_frac_min default fraction=0.5 minimum ----------')
    new_data = fillna_frac_min(dataset)
    print(new_data)

    print('\n--- keep_atleast min_samples=3 ----------')
    new_data = keep_atleast(dataset, min_samples=3)
    print(new_data)
    print('--- keep_atleast min_samples=5/6 ----------')
    new_data = keep_atleast(dataset, min_samples=5/6)
    print(new_data)
    print('\n--- keep_atleast_inlabels min_samples=2 ----------')
    new_data = keep_atleast_inlabels(dataset, min_samples=2)
    print(new_data)

    # read sample data set
    print('\nNormalization by reference feature ------------\n')
    print('---- original -------------------')
    data = dataset = dataio.read_data_csv(six.StringIO(datasets.demo_data2()), has_labels=True)
    print(dataset)
    print('------after normalizing by 97.59001 -----------------')
    new_data = normalize_ref_feature(data, 97.59001)
    print(new_data)

    # read sample data set
    print('\npareto scaling ------------\n')
    print('---- original -------------------')
    data = dataset = dataio.read_data_csv(six.StringIO(datasets.demo_data2()), has_labels=True)
    print(dataset)
    print('------after Pareto scaling -----------------')
    new_data = pareto_scale(data)
    print(new_data)
