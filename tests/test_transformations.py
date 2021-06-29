import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from io import StringIO
#from metabolinks import parse_data
#from metabolinks.dataio import gen_df, read_data_csv
from metabolinks.datasets import demo_dataset
from metabolinks import transformations as trans

# Sample data
# sample label
# s38    l1
# s39    l2
# s40    l1
# s32    l2
# s33    l2
# s34    l2

# 19 features

# m/z            97.58868   97.59001   97.59185  ...   98.57790   98.57899   99.28772
# label sample                                   ...
# l1    s38     1073218.0   637830.0   460092.0  ...  3891025.0        NaN  2038979.0
# l2    s39     1049440.0   534900.0   486631.0  ...  3990442.0  1877649.0        NaN
# l1    s40     1058971.0   582966.0        NaN  ...  3888258.0  1864650.0        NaN
# l2    s32     2351567.0  1440216.0  1137139.0  ...  2133404.0  1573559.0        NaN   
#       s33     1909877.0  1124346.0   926038.0  ...        NaN        NaN  3476845.0   
#       s34     2197036.0  1421899.0  1176756.0  ...  3643682.0  1829208.0        NaN

# [6 rows x 19 columns]

# -- Missing value counts support in () ----
#           global (6)  l1 (2)  l2 (4)
# m/z
# 97.58868           0       0       0
# 97.59001           0       0       0
# 97.59185           1       1       0
# 97.72992           4       1       3
# 98.34894           3       0       3
# 98.35078           3       0       3
# 98.35122           4       1       3
# 98.36001           3       1       2
# 98.57354           4       2       2
# 98.57382           3       1       2
# 98.57497           3       0       3
# 98.57528           3       2       1
# 98.57599           4       1       3
# 98.57621           3       1       2
# 98.57692           3       0       3
# 98.57712           3       2       1
# 98.57790           1       0       1
# 98.57899           2       1       1
# 99.28772           4       1       3

@pytest.fixture
def load_demo2():
    return demo_dataset('demo2')

@pytest.fixture
def data(load_demo2):
    return load_demo2.data

@pytest.fixture
def demo_target(load_demo2):
    return load_demo2.target

def test_fillna_zero(data):
    """test fillna_zero() function."""
    new_data = trans.fillna_zero(data)
    
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19)
    assert new_data.iloc[2, 2] == 0.0

def test_fillna_value(data):
    """test fillna_value() function"""
    new_data = trans.fillna_value(data, value=10.0)
    
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19)
    assert new_data.iloc[2, 2] == 10.0

def test_fillna_frac_min(data):
    """test fillna_frac_min() function."""
    globalmin = 0.5 * data.min().min()
    new_data = trans.fillna_frac_min(data, fraction=0.5)
    
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19)
    assert new_data.iloc[2, 2] == globalmin

def test_fillna_frac_min_feature(data):
    """test fillna_frac_min_feature() function."""
    minima = 0.2 * data.min()
    assert type(minima) == pd.Series
    new_data = trans.fillna_frac_min_feature(data, fraction=0.2)
    
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19)
    assert new_data.iloc[2, 2] == minima.iloc[2]
    assert new_data.iloc[0, 3] == minima.iloc[3]


def test_LODImputer_global_min(data):
    """test fillna_frac_min() function."""
    globalmin = 0.5 * data.min().min()

    tf = trans.LODImputer(strategy="global_min", fraction=0.5)
    new_data = tf.fit_transform(data)
    
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19)
    assert new_data.iloc[2, 2] == globalmin
    col2 = new_data.columns[2]
    tf.imputer_dict_[col2] == globalmin


def test_LODImputer_feature_min(data):
    """test LODIMputer Transformer with minimum per feature."""
    minima = 0.2 * data.min()
    assert type(minima) == pd.Series
    tf = trans.LODImputer(strategy="feature_min", fraction=0.2)
    new_data = tf.fit_transform(data)
    
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19)
    assert new_data.iloc[2, 2] == minima.iloc[2]
    assert new_data.iloc[0, 3] == minima.iloc[3]


def test_KeepMinimumNonNA_with_intmin(data):
    tf = trans.KeepMinimumNonNA(minimum=3)
    new_data = tf.fit_transform(data)
    predicted_ndropped = 5
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19-predicted_ndropped)
    assert tf.features_to_drop_[0] == data.columns[3]

def test_KeepMinimumNonNA_with_floatmin(data):
    tf = trans.KeepMinimumNonNA(minimum=5/6)
    new_data = tf.fit_transform(data)
    predicted_ndropped = 15
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19-predicted_ndropped)
    assert tf.features_to_drop_[0] == data.columns[3]
    assert tf.features_to_drop_[1] == data.columns[4]

def test_KeepMinimumNonNA_with_target(data, demo_target):
    tf = trans.KeepMinimumNonNA(minimum=2)
    assert len(pd.unique(demo_target)) == 2
    new_data = tf.fit_transform(data, demo_target)
    predicted_ndropped = 4
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19-predicted_ndropped)
    assert tf.features_to_drop_[0] == data.columns[3]

def test_keep_atleast_with_intmin(data):
    new_data = trans.keep_atleast(data, minimum=3)
    predicted_ndropped = 5
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19-predicted_ndropped)

def test_keep_atleast_with_floatmin(data):
    new_data = trans.keep_atleast(data, minimum=5/6)
    predicted_ndropped = 15
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19-predicted_ndropped)

def test_keep_atleast_with_intmin_and_target(data, demo_target):
    new_data = trans.keep_atleast(data, minimum=2, y=demo_target)
    predicted_ndropped = 4
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.shape == (6, 19-predicted_ndropped)

def test_norm_ref_feature(data):
    # 97.59001 is feature in pos 1
    tf = trans.SampleNormalizer(method='feature', feature=97.59001)
    new_data = tf.fit_transform(data)
    assert isinstance(new_data, pd.DataFrame)
    assert (new_data.iloc[:, 1].values == np.full(data.shape[0], 1.0)).all()
    assert (tf.scaling_factors_ == data.iloc[:, 1]).all()
    assert new_data.iloc[2, 0] == data.iloc[2, 0] / data.iloc[2, 1]

def test_norm_ref_feature_with_drop_features(data):
    # 97.59001 is feature in pos 1
    # 97.59185 is feature in pos 2
    tf = trans.SampleNormalizer(method='feature', feature=97.59001)
    f1 = 97.59001
    f2 = 97.59185
    closest = trans.find_closest_features(data, features=[f1, f2])
    assert closest[f1] is not None
    assert closest[f2] is not None
    
    new_data = tf.fit_transform(data)
    assert (tf.scaling_factors_ == data.iloc[:, 1]).all()
    
    tf2 = trans.DropFeatures(features_to_drop=[f1, f2])
    new_data = tf2.fit_transform(new_data)
    
    assert isinstance(new_data, pd.DataFrame)
    assert new_data.iloc[2, 0] == data.iloc[2, 0] / data.iloc[2, 1]
    kept_cols_mask = np.full(data.shape[1], True)
    kept_cols_mask[1:3] = (False, False)
    kept_cols = data.columns[kept_cols_mask]
    assert new_data.shape[1] == data.shape[1] - 2
    assert (kept_cols == new_data.columns).all()
    
    closest = trans.find_closest_features(new_data, features=[f1, f2])
    assert closest[f1] is None
    assert closest[f2] is None

def test_norm_total(data):
    tf = trans.SampleNormalizer(method='total')
    new_data = tf.fit_transform(data)
    assert isinstance(new_data, pd.DataFrame)
    totals = data.sum(axis=1).values
    assert (tf.scaling_factors_ == totals).all()
    assert new_data.iloc[2, 0] == data.iloc[2, 0] / data.iloc[2, :].sum()

def test_norm_PQN(data):
    # using mean as ref sample
    tf = trans.SampleNormalizer(method='PQN', ref_sample='mean')
    new_data = tf.fit_transform(data)
    assert isinstance(new_data, pd.DataFrame)
    mean_sample = data.mean(axis=0)
    sf_first_sample = (data.iloc[0, :] / mean_sample).median()
    assert tf.scaling_factors_[0] == sf_first_sample
    assert new_data.iloc[2, 0] == data.iloc[2, 0] / tf.scaling_factors_[2]
    
    # using first sample as ref
    tf = trans.SampleNormalizer(method='PQN', ref_sample=('l1', 's38'))
    new_data = tf.fit_transform(data)
    assert isinstance(new_data, pd.DataFrame)
    ref_sample = data.iloc[0, :]
    sf_second_sample = (data.iloc[1, :] / ref_sample).median()
    assert tf.scaling_factors_[1] == sf_second_sample
    assert new_data.iloc[2, 0] == data.iloc[2, 0] / tf.scaling_factors_[2]

def test_glog(data):
    # this test requires missing value imputation
    # to be done before transformation
    # global 1/2 minimum is used as imputation
    tf = trans.LODImputer(strategy="global_min", fraction=0.5)
    imputed = tf.fit_transform(data)
    tf = trans.GLogTransformer()
    new_data = tf.fit_transform(imputed)
    # just a few straight comparisons
    # following transform() code
    lamb = imputed.min().min()/10.0
    y = imputed.values
    y = np.log2((y + (y**2 + lamb**2)**0.5)/2)
    assert isinstance(new_data, pd.DataFrame)
    assert (new_data.values == y).all()

def test_pareto(data):
    tf = trans.FeatureScaler(method='pareto')
    new_data = tf.fit_transform(data)
    assert isinstance(new_data, pd.DataFrame)
    # just a few straight comparisons
    # following transform() code
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    scaled = data.sub(means, axis=1).div(stds**0.5, axis=1)
    assert_frame_equal(new_data, scaled)

def test_standard(data):
    tf = trans.FeatureScaler(method='standard')
    new_data = tf.fit_transform(data)
    assert isinstance(new_data, pd.DataFrame)
    # just a few straight comparisons
    # following transform() code
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    scaled = data.sub(means, axis=1).div(stds, axis=1)
    assert_frame_equal(new_data, scaled)

def test_range(data):
    tf = trans.FeatureScaler(method='range')
    new_data = tf.fit_transform(data)
    assert isinstance(new_data, pd.DataFrame)
    # just a few straight comparisons
    # following transform() code
    means = data.mean(axis=0)
    maxima = data.max(axis=0)
    minima = data.min(axis=0)
    ranges = maxima - minima
    scaled = data.sub(means, axis=1).div(ranges, axis=1)
    assert_frame_equal(new_data, scaled)

def test_mean_center(data):
    tf = trans.FeatureScaler(method='mean_center')
    new_data = tf.fit_transform(data)
    assert isinstance(new_data, pd.DataFrame)
    # just a few straight comparisons
    # following transform() code
    means = data.mean(axis=0)
    scaled = data.sub(means, axis=1)
    assert_frame_equal(new_data, scaled)

def test_vast(data):
    tf = trans.FeatureScaler(method='vast')
    new_data = tf.fit_transform(data)
    assert isinstance(new_data, pd.DataFrame)
    # just a few straight comparisons
    # following transform() code
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    scaled = data.sub(means, axis=1)
    scaled = scaled.div(stds, axis=1)
    scaled = scaled.mul(means, axis=1)
    scaled = scaled.div(stds, axis=1)
    assert_frame_equal(new_data, scaled)

def test_level_mean(data):
    tf = trans.FeatureScaler(method='level_mean')
    new_data = tf.fit_transform(data)
    assert isinstance(new_data, pd.DataFrame)
    # just a few straight comparisons
    # following transform() code
    means = data.mean(axis=0)
    scaled = data.sub(means, axis=1)
    scaled = scaled.div(means, axis=1)
    assert_frame_equal(new_data, scaled)

def test_level_median(data):
    tf = trans.FeatureScaler(method='level_median')
    new_data = tf.fit_transform(data)
    assert isinstance(new_data, pd.DataFrame)
    # just a few straight comparisons
    # following transform() code
    means = data.mean(axis=0)
    medians = data.median(axis=0)
    scaled = data.sub(means, axis=1)
    scaled = scaled.div(medians, axis=1)
    assert_frame_equal(new_data, scaled)
