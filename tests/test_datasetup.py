import pytest
import numpy as np
import pandas as pd
from metabolinks.datasetup import gen_df

def assert_almost_equal(x, y):
    if abs(x-y) < 0.0001:
        return True
    return False

def test_setup_from_np_array():
    """test construction from numpy array"""
    n = 12
    nrows = 4
    data = np.array(range(12 * 4)).reshape(nrows, n)
    df = gen_df(data)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4,12)
    assert len(df.index.names) == 1
    assert len(df.columns.names) == 2
    assert df.index.names[0] is None
    assert df.columns.names == ['label', 'sample']
    assert df.columns.levels[0] == ['no label']
    assert df.columns.get_level_values(1)[2] == 'Sample 3'
    assert df.values[1,2] == 14

if __name__ == '__main__':
    pytest.main()