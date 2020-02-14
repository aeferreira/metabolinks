"""An accessor to Pandas DataFrame to interpret content as MS data."""

import numpy as np
import pandas as pd

from pandas_flavor import register_dataframe_accessor

from .utils import _is_string

@register_dataframe_accessor("ms")
class MSAccessor(object):
    """An accessor to Pandas DataFrame to interpret content as MS data.

    The following convention is enforced for the DataFrame:

    Direct metadata is stored in the (hierarquical) column index.
    This index contain three levels (outwards from the data):

    - `info_type` describes the type of data in each column. The default is "I". A common example is
       the pair "RT", "I" for each sample
    - `sample` contains the names of samples
    - `label` contains the labels of each sample

    The (row) index contains one level, the "features", often labels of spectral entities. Examples are
    m/z values, formulae or any format-specific labeling scheme.
    """

    def __init__(self, df):
        self._validate(df)
        self._df = df
    
    @staticmethod
    def _validate(df):
        if not isinstance(df, pd.DataFrame):
            raise AttributeError("'ms' must be used with a Pandas DataFrame")
        if len(df.columns.names) < 2:
            raise AttributeError('Must have at least label and sample metadata on columns')

    @property
    def data_matrix(self):
        """The Pandas DataFrame holding the MS data, transposed to be usable as tidy"""
        return self._df.transpose(copy=True)

    def _get_zip_labels_samples(self):
        return zip(self._df.columns.get_level_values(0), self._df.columns.get_level_values(1))

    @property
    def labels(self):
        """Get the different data labels (no repetitions)."""
        return tuple(self._df.columns.levels[0])

    @labels.setter
    def labels(self, value):
        self._rebuild_col_level(value, 0)

    @property
    def iterlabels(self):
        return self._df.columns.get_level_values(0)

    @property
    def samples(self):
        """Get the different sample names."""
        return tuple(self._df.columns.levels[1])

    @samples.setter
    def samples(self, value):
        self._rebuild_col_level(value, 1)

    def _rebuild_col_level(self, value, level):
        cols = self._df.columns
        n = len(cols)
        metanames = cols.names
        # handle value
        if value is None or len(value) == 0:
            if level == 0:
                value = ['no label']
            elif level == 1:
                value = [f'Sample {i}' for i in range(1, n + 1)]
            else:
                value = [f'Info {i}' for i in range(1, n + 1)]
        elif _is_string(value):
            value = [value]
        else:
            value = list(value)
        nr = n // len(value)
        newstrs = []
        for s in value:
            newstrs.extend([s]*nr)
        cols = [list(c) for c in cols]
        for i, s in enumerate(newstrs):
            cols[i][level] = s
        newcols = [tuple(c) for c in cols]
        self._df.columns = pd.MultiIndex.from_tuples(newcols, names=metanames)

    @property
    def itersamples(self):
        return self._df.columns.get_level_values(1)

    @property
    def feature_count(self):
        """Get the number of features."""
        return len(self._df.index)

    @property
    def sample_count(self):
        """Get the number of samples."""
        return len(self.samples)

    @property
    def iter_labels_samples(self):
        return self._get_zip_labels_samples()

    @property
    def label_count(self):
        """Get the number of labels."""
        # 'no label' still counts as one (global) label
        return len(self.labels)

    @property
    def no_labels(self):
        """True if there is only one (global) label 'no label'."""
        return self.label_count == 1 and self.labels[0] == 'no label'

    @property
    def info_types(self):
        ncl = len(self._df.columns.names)
        if ncl == 3:
            return tuple(self._df.columns.levels[3])
        else:
            return tuple()

    def info(self, all_data=False):
        if all_data:
            dfres = [('samples', self.sample_count),
                     ('labels', self.label_count),
                     ('features', self.feature_count)]
            return dict(dfres)
        ls_table = [(s,l) for (l,s) in self._get_zip_labels_samples()]
        ls_table.append((self.sample_count, self.label_count))
        indx_strs = [str(i) for i in range(self.sample_count)] + ['global']
        return pd.DataFrame(ls_table, columns=['sample', 'label'], index=indx_strs)

    def label_of(self, sample):
        """Get label from sample name"""
        for lbl, s in self._get_zip_labels_samples():
            if s == sample:
                return lbl
        raise KeyError(f"No label found for '{sample}'")

    def samples_of(self, label):
        """Get a list of sample names from label"""
        snames = [lbl for s, lbl in self._get_zip_labels_samples() if lbl == label]
        return snames

    def _get_subset_data(self, sample=None, info=None, label=None, no_drop_na=False):
        if sample is None and info is None and label is None:
            return self._df.copy()
        if sample is not None:
            if _is_string(sample):
                samples = [sample]
            else:
                samples = list(sample)
            indexer = []
            for s in samples:
                if s not in self.samples:
                    raise KeyError(f"'{s}' is not a sample name")
                lbl = self.label_of(s)
                indexer.append((lbl, s))
            if len(indexer) == 1:
                indexer = indexer[0]
            df = self._df.loc[:, indexer]
        elif sample is None and label is not None:
            if _is_string(label):
                labels = [label]
            else:
                labels = list(label)
            indexer = []
            for s in labels:
                if s not in self.labels:
                    raise KeyError(f"'{s}' is not a sample name")
                indexer.append(s)
            df = self._df.loc[:, (indexer,)]
        else:
            raise KeyError("Sample name or label not found")
        if no_drop_na:
            df = df.copy()
        else:
            df = df.dropna(how='all')
        if isinstance(df, pd.DataFrame):
            df.columns = df.columns.remove_unused_levels()
        return df

    def take(self, **kwargs):
        return self._get_subset_data(**kwargs)

    def restrict(self, **kwargs):
        return self.take(**kwargs)

    def features(self, **kwargs):
        df = self._get_subset_data(**kwargs)
        return df.index

    def transform(self, func, drop_na=True, **kwargs):
        df = self._df
        df = df.pipe(func, **kwargs)
        if drop_na:
            df = df.dropna(how='all')
        if isinstance(df, pd.DataFrame):
            df.columns = df.columns.remove_unused_levels()
        return df
