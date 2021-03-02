"""All transformations should accept a pandas DataFrame object.

   Most should return a new pandas DataFrame.
   The input data matrix should follow the convention that the
   instances are in rows and features are in columns.
"""
from functools import partial
from typing import Optional, List, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_input_matches_training_df,
)
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
    _find_all_variables,
)
from feature_engine.imputation import ArbitraryNumberImputer
from feature_engine.wrappers import SklearnTransformerWrapper


import metabolinks as mtls
from metabolinks.utils import _is_string
Variables = Union[None, int, str, List[Union[str, int]]]

# ---------- imputation of missing values -------

def fillna_zero(df):
    """Set NaN to zero."""
    return ArbitraryNumberImputer(arbitrary_number=0.0).fit_transform(df)

def fillna_value(df, value=0.0):
    """Set NaN to value."""
    return ArbitraryNumberImputer(arbitrary_number=value).fit_transform(df)
    #return df.fillna(value)

def fillna_frac_min(df, fraction=0.5):
    """Set NaN to a fraction of the minimum value in whole DataFrame."""

    # minimum = df.min().min()
    # # print(minimum)
    # minimum = minimum * fraction
    return FracMinImputer(fraction=fraction).fit_transform(df)

class FracMinImputer(BaseImputer):
    """
    The FracMinImputer() transforms features by replacing missing data by a fraction
    of the global minimum value of a pandas DataFrame.

    Parameters
    ----------
    fraction : float, default=0.5
        Factor to multiply the minimum.
    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all variables of type numeric.
    Attributes
    ----------
    imputer_dict_:
        Dictionary with values equal to the constant replacement value.
    imputation_value_:
        The value to used to replace the NAs.
    Methods
    -------
    fit:
        Learn values to replace missing data.
    transform:
        Impute missing data.
    fit_transform:
        Fit to the data, then transform it.
    """

    def __init__(self, fraction: float = 0.5, variables: Variables = None,) -> None:

        if fraction <= 0:
            raise ValueError("fraction must be a positive number")

        self.fraction = fraction
        self.variables = _check_input_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the value of the fraction of the minimum of the data.
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training dataset.
        y : pandas Series, default=None
            y is not needed in this imputation. You can pass None or y.
        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame
            - If any of the user provided variables are not numerical
        ValueError
            If there are no numerical variables in the df or the df is empty
        Returns
        -------
        self
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables = _find_or_check_numerical_variables(X, self.variables)

        # estimate imputation value
        minimum = X.min().min()
        # print(minimum)
        minimum = minimum * self.fraction

        self.imputer_dict_ = {c: minimum for c in self.variables}
        self.imputation_value_ = minimum

        self.input_shape_ = X.shape

        return self

    # # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    # def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    #     X = super().transform(X)

    #     return X

    # transform.__doc__ = BaseImputer.transform.__doc__

# ---------- filters for reproducibility

class KeepMinimumNonNA(BaseSelector):
    """
    Keep variables from a dataframe when the number of non-missing values exceeds
    a minimum threshold.
    If minimum is an int, it represents a minimum number of feature ocorrences in samples.
    If 0 <= minimum < 1, it represents the minimum number of feature ocorrences
    as a fraction of the number of samples.
    The number of samples can be the global number of samples or the number of samples
    in each groups of samples if a target `y` containing labels indicating group
    membership is provided as an argument of `fit`.

    This transformer works with both numerical and categorical variables.
    The user can indicate a list of variables to examine.
    Alternatively, the transformer will evaluate all the variables in the dataset.
    The transformer will first identify and store the features with too many missing values (fit).
    Next, the transformer will drop these variables from a dataframe (transform).
    Parameters
    ----------
    minimum : float,int,  default=1
        Threshold of non-NA ocorrences for a feature to be kept. If int,
        it represents the minimum of samples. If 0 <= tol < 1, it represents
        the minimum ocorrence of non-missing values as a fraction of the number of samples.
        Number of samples is the global number of samples if a target `y=None` or
        the number of samples in each group if a target `y`with labels indicating
        gorup membership is provided as an argument of `fit`.
    variables : list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        variables in the dataset.
    Attributes
    ----------
    features_to_drop_:
        List with features with too many missing values.
    Methods
    -------
    fit:
        Find features with too many missing values.
    transform:
        Remove features with too many missing values.
    fit_transform:
        Fit to the data. Then transform it.
    """

    def __init__(self, minimum: float = 1, variables: Variables = None,):

        if not isinstance(minimum, (float, int)) or minimum < 0:
            raise ValueError("minimum must be an integer or a float between 0 and 1")

        self.minimum = minimum
        self.variables = _check_input_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Find constant and quasi-constant features.
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input dataframe.
        y : array-like of shape (n_samples)
           Target describing group membership of samples. None to use global number of samples.
        Returns
        -------
        self
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find all variables or check those entered are present in the dataframe
        self.variables = _find_all_variables(X, self.variables)

        if y is None:
            counts = X.count(axis=0)
            # a float in (0,1) means a fraction
            tol = self.minimum
            if 0.0 < tol < 1.0:
                n = X.shape[0]
                tol = tol * float(n)
            self.features_to_drop_ = list(X.columns[counts < tol])
        else:
            unique_labels = pd.unique(y)
            keep_dict = {}
            tol = self.minimum
            for lbl in unique_labels:
                X_lbl = X[y == lbl]
                if 0.0 < tol < 1.0:
                    n = X_lbl.shape[0]
                    tol = tol * float(n)
                counts = X_lbl.count(axis=0)
                keep_dict[lbl] = counts >= tol
            all_keeps = pd.DataFrame(keep_dict).transpose()
            # print('all_keeps dataframe')
            # print(all_keeps.transpose())
            keep_vars = all_keeps.any(axis=0)
            # print('keep_vars')
            # print(keep_vars)
            self.features_to_drop_ = list(keep_vars[keep_vars==False].index)

        # check we are not dropping all the features
        if len(self.features_to_drop_) == len(X.columns):
            raise ValueError(
                "The resulting dataframe will have no columns after dropping all "
                "features with too many missing values. Try changing the minimum value."
            )

        self.input_shape_ = X.shape

        return self

    # # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    # def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    #     X = super().transform(X)

    #     return X

    # transform.__doc__ = BaseSelector.transform.__doc__

def keep_atleast(df, minimum=1, y=None):
    """Keep only features which occur at least `minimum` times in samples.

       If 0 < min_samples < 1, this should be interpreted as a fraction of the number of samples.
       If target `y` is provided, the number of samples to compute the ocorrence is the number of
       samples in each group.
       """

    tf = KeepMinimumNonNA(variables=None, minimum=minimum)
    return tf.fit_transform(df, y=y)

# ---------- normalizations -----------------------------------

# A bit of nomenclature: "normalization" is used here to denote transformations that adjust
# data to correct or compensate errors relating to overall diferences in sample concentrations
# (for example, as resulting from different sample dilutions).
# The assumption is that tere are factors that affect the majority of features in a systematic
# way (as oposed to differences in features )
# It does not necessarily produce samples with unit norms, as in the Normalizer of
# scikit-learn

# this transformer 


class RefFeatureNormalizer(BaseEstimator, TransformerMixin):
    """
    The RefFeatureNormalizer() divides all numerical features of a
    dataframe by the values of a reference column, a 'reference feature'.
    NA values are kept as NA values.
    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all variables of type numeric.
    Parameters
    ----------
    feature: string, float
        Indicates the feature to use for normalization.
    variables : list, default=None
        The list of numerical variables to be transformed. If None, the transformer
        will find and select all numerical variables.
    fold: float, default=1.0
        After division by the reference feature values, the dataframe is then multiplied by fold.
    Attributes
    ----------
    ref_features_values_:
        pandas Series with the reference feature values.
    Methods
    -------
    fit:
        fill the ref_features_values_.
    transform:
        Transforms the variables by dividing from the reference feature. Then, multiply by fold.
    fit_transform:
        Fit to data, then transform it.
    """

    def __init__(
        self,
        feature: Union[str, float],
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        fold: Union[str, float] = 1.0,
    ) -> None:

        self.feature = feature
        self.variables = _check_input_parameter_variables(variables)
        self.fold = fold

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Checks that input is a dataframe, finds numerical variables, or alternatively
        checks that variables entered by the user are of type numerical.
        Parameters
        ----------
        X : Pandas DataFrame
        y : Pandas Series, np.array. Default = None
            Parameter is necessary for compatibility with sklearn.pipeline.Pipeline.
        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
            If any of the user provided variables are not numerical
        ValueError
            If there are no numerical variables in the df or the df is empty
        Returns
        -------
        X : Pandas DataFrame
            The same dataframe entered as parameter
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables: List[Union[str, int]] = _find_or_check_numerical_variables(
            X, self.variables
        )
        new_index, indexer = X.columns.sort_values(return_indexer=True)
        pos = new_index.get_loc(self.feature, method='pad')
        pos = indexer[pos]

        self.ref_features_values_ = X.iloc[:, pos]

        self.input_shape_ = X.shape
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Divide dataframe by reference feature column.
        Parameters
        ----------
        X : Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.
        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the dataframe not of the same size as that used in fit().
        Returns
        -------
        X : pandas dataframe
            The dataframe with the transformed variables.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_input_matches_training_df(X, self.input_shape_[1])

        # transform
        X[self.variables] = X[self.variables].div(self.ref_features_values_, axis=0)
        return X

class DropFeatures(BaseSelector):
    """
    DropFeatures() drops a list of variable(s) indicated by the user from the dataframe.
    This is just the DropFeatures transformer from feature-engine, adapted to locate
    features as floats.

    Parameters
    ----------
    features_to_drop : str or list, default=None
        Variable(s) to be dropped from the dataframe
    Methods
    -------
    fit:
        This transformer does not learn any parameter.
    transform:
        Drops indicated features.
    fit_transform:
        Fit to data, then transform it.
    """

    def __init__(self, features_to_drop: List[Union[str, int]]):

        if not isinstance(features_to_drop, list) or len(features_to_drop) == 0:
            raise ValueError(
                "features_to_drop should be a list with the name of the variables"
                "you wish to drop from the dataframe."
            )

        self.features_to_drop = features_to_drop

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        This transformer does not learn any parameter.
        Verifies that the input X is a pandas dataframe, and that the variables to
        drop exist in the training dataframe.
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input dataframe
        y : pandas Series, default = None
            y is not needed for this transformer. You can pass y or None.
        Returns
        -------
        self
        """
        # check input dataframe
        X = _is_dataframe(X)
        self.features_to_drop_ = []

        # find position of feature
        new_index, indexer = X.columns.sort_values(return_indexer=True)
        for feature in self.features_to_drop:
            pos = new_index.get_loc(feature, method='pad')
            pos = indexer[pos]
            self.features_to_drop_.append(X.columns[pos])

        # check user is not removing all columns in the dataframe
        if len(self.features_to_drop) == len(X.columns):
            raise ValueError(
                "The resulting dataframe will have no columns after dropping all "
                "existing variables"
            )

        # add input shape
        self.input_shape_ = X.shape

        return self

    # # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    # def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    #     X = super().transform(X)

    #     return X

    # transform.__doc__ = BaseSelector.transform.__doc__

def normalize_ref_feature(df, feature, remove=False):
    """Normalize dataset by a reference feature (a column label).

       df: a Pandas DataFrame.
       feature: column label.
       remove: bool; True to remove reference feature from data after normalization.

       Returns: DataFrame.
       """

    # find position of feature
    new_df = RefFeatureNormalizer(feature=feature).fit_transform(df)
    if remove:
        new_df = DropFeatures(features_to_drop=[feature]).fit_transform(new_df)
    return(df)


def drop_features(df, features):
    """Remove a reference feature (an exact row label).

       df: a Pandas DataFrame.
       features: list of column names.

       Returns: DataFrame.
       """

    return DropFeatures(features_to_drop=[features]).fit_transform(df)

def normalize_sum(df):
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
    
    #"Building" the reference sample based on the input given
    if ref_sample == 'mean': #Mean spectre of all samples
        ref_sample2 = df / df.mean()
    elif ref_sample == 'median': #Median spectre of all samples
        ref_sample2 = df/df.median()
    elif ref_sample in df.index: # Sample name to use as a reference
        ref_sample2 = df/df.loc[ref_sample,:]
    else: # Actual sample given (ref_sample is array like)
        ref_sample2 = df/ref_sample
    #Normalization Factor and Normalization
    pqr_fact = ref_sample2.median()
    return df / pqr_fact


class PQNormalizer(BaseNumericalTransformer):
    """
    The PQNormalizer() divides all numerical features of a
    dataframe by the values of a reference column, a 'reference feature'.
    NA values are kept as NA values.
    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all variables of type numeric.
    Parameters
    ----------
    feature: string, float
        Indicates the feature to use for normalization.
    variables : list, default=None
        The list of numerical variables to be transformed. If None, the transformer
        will find and select all numerical variables.
    fold: float, default=1.0
        After division by the reference feature values, the dataframe is then multiplied by fold.
    Attributes
    ----------
    ref_features_values_:
        pandas Series with the reference feature values.
    Methods
    -------
    fit:
        fill the ref_features_values_.
    transform:
        Transforms the variables by dividing from the reference feature. Then, multiply by fold.
    fit_transform:
        Fit to data, then transform it.
    """

    def __init__(
        self,
        feature: Union[str, float],
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        fold: Union[str, float] = 1.0,
    ) -> None:

        self.feature = feature
        self.variables = _check_input_parameter_variables(variables)
        self.fold = fold

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Checks that input is a dataframe, finds numerical variables, or alternatively
        checks that variables entered by the user are of type numerical.
        Parameters
        ----------
        X : Pandas DataFrame
        y : Pandas Series, np.array. Default = None
            Parameter is necessary for compatibility with sklearn.pipeline.Pipeline.
        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
            If any of the user provided variables are not numerical
        ValueError
            If there are no numerical variables in the df or the df is empty
        Returns
        -------
        X : Pandas DataFrame
            The same dataframe entered as parameter
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables: List[Union[str, int]] = _find_or_check_numerical_variables(
            X, self.variables
        )
        new_index, indexer = X.columns.sort_values(return_indexer=True)
        pos = new_index.get_loc(self.feature, method='pad')
        pos = indexer[pos]

        self.ref_features_values_ = X.iloc[:, pos]

        self.input_shape_ = X.shape
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Divide dataframe by reference feature column.
        Parameters
        ----------
        X : Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.
        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the dataframe not of the same size as that used in fit().
        Returns
        -------
        X : pandas dataframe
            The dataframe with the transformed variables.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_input_matches_training_df(X, self.input_shape_[1])

        # transform
        X[self.variables] = X[self.variables].div(self.ref_features_values_, axis=0)
        return X

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

    #Replacing the values in the dataset for the reference sample values based on the ranks calculated earlier for each entry
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
    demo2 = demo_dataset('demo2')
    data = demo2.data
    y = demo2.target
    print('-- info --------------')
    print(data.transpose().cdl.info())
    print('-- global info---------')
    print(data.transpose().cdl.info(all_data=True))
    print('-----------------------')
    print(data)

    print('\n--- fillna_zero ----------')
    new_data = fillna_zero(data)
    print(new_data)
    print('--- fillna_value  10 ----------')
    new_data = fillna_value(data, value=10)
    print(new_data)
    print('--- fillna_frac_min default fraction=0.5 minimum ----------')
    new_data = fillna_frac_min(data)
    print(new_data)
    print('--- fillna_frac_min default fraction=0.5 minimum with FraMinImputer----------')
    tf = FracMinImputer(fraction=0.5)
    new_data = tf.fit_transform(data)
    print(new_data)
    print('++'*20)
    print(tf.imputation_value_)

    print('\n--- keep at least minimum=3 using KeepMinNonNA----------')
    tf = KeepMinimumNonNA(minimum=3)
    new_data = tf.fit_transform(data)
    print(new_data)
    print('Dropped:', tf.features_to_drop_)
    print('\n--- keep at least minimum=5/6 using KeepMinNonNA----------')
    tf = KeepMinimumNonNA(minimum=5/6)
    new_data = tf.fit_transform(data)
    print(new_data)
    print('Dropped:', tf.features_to_drop_)

    print('\n--- keep at least minimum=2 using KeepMinNonNA with "target"----------')
    print('target is a label (list like) -------------------------')
    tf = KeepMinimumNonNA(minimum=2)
    print('target:', y)
    print(type(y))
    print('unique_labels:', pd.unique(y))
    labels = np.array(demo2.target_names)[y]
    print('labels:', labels)
    new_data = tf.fit_transform(data, y)
    print(new_data)
    print('Dropped:', tf.features_to_drop_)
    print('\n--- keep_atleast minimum=3 ----------')
    new_data = keep_atleast(data, minimum=3)
    print(new_data)
    print('--- keep_atleast minimum=5/6 ----------')
    new_data = keep_atleast(data, minimum=5/6)
    print(new_data)
    print('\n--- keep_atleast minimum=2 with target ----------')
    new_data = keep_atleast(data, minimum=2, y=y)
    print(new_data)

    print('\nNormalization by reference feature ------------\n')
    print('---- original -------------------')
    print(data)
    print('------after normalizing by 97.59001 -----------------')
    tf = RefFeatureNormalizer(feature=97.59001)
    new_data = tf.fit_transform(data)
    print(new_data)
    print('------after normalizing by 97.59001 and dropping that feature and also 97.59185---')
    tf = RefFeatureNormalizer(feature=97.59001)
    new_data = tf.fit_transform(data)
    tf = DropFeatures(features_to_drop=[97.59001, 97.59185])
    new_data = tf.fit_transform(new_data)
    print(new_data)

    # read sample data set
    print('\npareto scaling ------------\n')
    print('---- original -------------------')
    print(data)
    print('------after Pareto scaling -----------------')
    new_data = pareto_scale(data)
    print(new_data)
