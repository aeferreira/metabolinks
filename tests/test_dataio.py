import pytest
import numpy as np
import pandas as pd
import six
from metabolinks.dataio import gen_df, read_data_csv
import metabolinks.datasets as datasets


def test_setup_from_np_array():
    """test construction from numpy array"""
    n = 12
    nrows = 4
    data = np.array(range(12 * 4)).reshape(nrows, n)
    df = gen_df(data)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4,12)
    assert len(df.index.names) == 1
    assert len(df.columns.names) == 1
    assert df.index.names[0] is None
    assert df.columns.names[0] is None
    assert df.values[1,2] == 14


def test_setup_from_np_array_with_labels():
    """test construction from numpy array"""
    n = 12
    nrows = 4
    data = np.array(range(12 * 4)).reshape(nrows, n)
    df = gen_df(data, add_labels=['exp1', 'exp2', 'exp3', 'exp4'])
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4,12)
    assert len(df.index.names) == 1
    assert len(df.columns.names) == 2
    assert df.index.names[0] is None
    assert len(df.columns.get_level_values(0)) == 12
    assert df.columns.names[0] == 'label'
    assert len(df.columns.levels[0]) == 4
    assert df.columns.get_level_values(0)[3] == 'exp2'
    assert df.values[1,2] == 14

def test_setup_from_DataFrame():
    """test construction from pandas DataFrame"""
    n = 12
    nrows = 4
    data = np.array(range(12 * 4)).reshape(nrows, n)
    exp_data = pd.DataFrame(
        data,
        index=['exp1', 'exp2', 'exp3', 'exp4'],
        columns=[f'SE{i}' for i in range(1, 13)],
    )
    df = gen_df(exp_data)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4,12)
    assert len(df.index.names) == 1
    assert len(df.columns.names) == 1
    assert df.index.names[0] is None
    assert df.columns.names[0] is None
    assert df.values[1,2] == 14
    assert df.columns[3] == 'SE4'

def test_read_csv_nolabels():
    """test construction from demo1 data set (CSV with no labels)."""
    d = datasets.demo_dataset('demo1').as_str()
    df = read_data_csv(six.StringIO(d))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (19,6)
    assert len(df.index.names) == 1
    assert len(df.columns.names) == 1
    assert df.index.names[0] == 'm/z'
    assert df.columns.names[0] is None
    assert df.values[1,2] == 582966
    assert df.columns[3] == 's32'
    assert df.isna().sum().sum() == 51

if __name__ == '__main__':
    pytest.main()