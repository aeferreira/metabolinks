"""Accessors to Pandas DataFrame to interpret content as MS data.

   Two versions: 'ms' assumes labeled data, 'ums' assumes unlabeled data"""

import numpy as np
import pandas as pd

from pandas_flavor import register_dataframe_accessor

from .utils import _is_string


def create_multiindex_with_labels(df, labels=["no label"], level_name="label"):
    cols = df.columns
    n = len(cols)
    metanames = cols.names
    if not labels:
        labels = ["no label"]
    elif _is_string(labels):
        labels = [labels]
    else:
        labels = list(labels)
    nr = n // len(labels)
    newstrs = []
    for s in labels:
        newstrs.extend([s] * nr)
    if len(metanames) > 1:
        tcols = [list(c) for c in cols.to_flat_index()]
    else:
        tcols = [[c] for c in cols]
    newcols = [tuple([ns] + c) for (ns, c) in zip(newstrs, tcols)]
    return pd.MultiIndex.from_tuples(newcols, names=[level_name] + metanames)


@register_dataframe_accessor("ms")
class MSAccessor(object):
    """An accessor to Pandas DataFrame to interpret content as MS data.

    This interpretation assumes that the **column** index stores the essential
    metadata, namely, sample names and group labels. This index is
    ususally hierarquical and other levels are optional. Accessor 'ums' for unlabeled data
    where level 0 are assumed to be sample names is also available in this module.

    Interpretation is based on the following conventions :

    - For the accessor to work, the column index must have at leat two levels.

    - Level 1 is interpreted as sample names by the accessor.
      Default name for this level is 'sample' and default values are 'Sample {i}'.
      .samples is a property to access this level.

    - Level 0 is interpreted as labels. .labels is a property to access labels.

    - More levels are possible, if they are read from data sources or added by Pandas index manipulation.

    The (row) index is interpreted as "features", often labels of spectral entities. Examples are
    m/z values, formulae or any format-specific labeling scheme. It may be hierarquical.
    """

    def __init__(self, df):
        self._validate(df)
        self._df = df

    @staticmethod
    def _validate(df):
        """Require a pandas DataFrame with at least two levels in column MultiIndex to work."""
        if not isinstance(df, pd.DataFrame):
            raise AttributeError("'ms' must be used with a Pandas DataFrame")
        if len(df.columns.names) < 2:
            raise AttributeError(
                "Must have at least label and sample metadata on columns"
            )

    @property
    def data_matrix(self):
        """The Pandas DataFrame holding the data, transposed to be usable as tidy"""
        return self._df.transpose(copy=True)

    def _get_zip_labels_samples(self):
        self._df.columns = self._df.columns.remove_unused_levels()
        return zip(
            self._df.columns.get_level_values(0), self._df.columns.get_level_values(1)
        )

    @property
    def labels(self):
        """Get the different data labels (with no repetitions)."""
        return tuple(self._df.columns.levels[0])

    @labels.setter
    def labels(self, value):
        """Setter for data labels."""
        self._rebuild_col_level(value, 0)

    @property
    def iterlabels(self):
        """iterate over labels of each DataFrame column."""
        self._df.columns = self._df.columns.remove_unused_levels()
        return self._df.columns.get_level_values(0)

    @property
    def samples(self):
        """Get the different sample names (with no repetitions in case the number of levels > 2)."""
        return tuple(self._df.columns.levels[1])

    @samples.setter
    def samples(self, value):
        """Setter for sample names."""
        self._rebuild_col_level(value, 1)

    def _rebuild_col_level(self, value, level):
        cols = self._df.columns.remove_unused_levels()
        n = len(cols)
        metanames = cols.names
        # handle value
        if value is None or len(value) == 0:
            if level == 0:
                value = ["no label"]
            elif level == 1:
                value = [f"Sample {i}" for i in range(1, n + 1)]
            else:
                value = [f"Info {i}" for i in range(1, n + 1)]
        elif _is_string(value):
            value = [value]
        else:
            value = list(value)
        nr = n // len(value)
        newstrs = []
        for s in value:
            newstrs.extend([s] * nr)
        cols = [list(c) for c in cols]
        for i, s in enumerate(newstrs):
            cols[i][level] = s
        newcols = [tuple(c) for c in cols]
        self._df.columns = pd.MultiIndex.from_tuples(newcols, names=metanames)

    @property
    def itersamples(self):
        """iterate over sample names of each DataFrame column."""
        self._df.columns = self._df.columns.remove_unused_levels()
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
        """iterate over pairs of (label, sample name) for each DataFrame column."""
        self._df.columns = self._df.columns.remove_unused_levels()
        return self._get_zip_labels_samples()

    @property
    def label_count(self):
        """Get the number of labels."""
        # 'no label' still counts as one (global) label
        return len(self.labels)

    @property
    def no_labels(self):
        """True if there is only one (global) label 'no label'."""
        return self.label_count == 1 and self.labels[0] == "no label"

    def info(self, all_data=False):
        """A dicionary of global counts or a DataFrame with info for each sample"""
        if all_data:
            return dict(
                samples=self.sample_count,
                labels=self.label_count,
                features=self.feature_count,
            )
        ls_table = [(s, l) for (l, s) in self._get_zip_labels_samples()]
        ls_table.append((self.sample_count, self.label_count))
        indx_strs = [str(i) for i in range(self.sample_count)] + ["global"]
        return pd.DataFrame(ls_table, columns=["sample", "label"], index=indx_strs)

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
            df = df.dropna(how="all")
        if isinstance(df, pd.DataFrame):
            df.columns = df.columns.remove_unused_levels()
        return df

    def take(self, **kwargs):
        """The function that indexes data by sample name or label."""
        return self._get_subset_data(**kwargs)

    def subset(self, **kwargs):
        """Alias for take()."""
        return self.take(**kwargs)

    def features(self, **kwargs):
        """Get the row index (features) indexing data by sample name or label"""
        df = self._get_subset_data(**kwargs)
        return df.index

    def pipe(self, func, drop_na=True, **kwargs):
        """Thin wrapper arounf DataFrame.pipe() with automatic dropna and housekeeping."""
        df = self._df
        df = df.pipe(func, **kwargs)
        if drop_na:
            df = df.dropna(how="all")
        if isinstance(df, pd.DataFrame):
            df.columns = df.columns.remove_unused_levels()
        return df

    def erase_labels(self):
        """Erase the labels level (level 0) in the column MultiIndex.

           CAUTION: the accessor will no longer work or misinterpret levels.
           After the application of this function use accessor ums afterwards."""

        self._df.columns = self._df.columns.droplevel(level=0)
        if len(self._df.columns.names) > 1:
            self._df.columns = self._df.columns.remove_unused_levels()
        return self._df.copy()


@register_dataframe_accessor("ums")
class UMSAccessor(object):
    """An accessor to Pandas DataFrame to interpret content as MS data.

    This interpretation assumes that the **column** index stores the sample names
    on level 0. This index is optionally hierarquical. Accessor 'ms' for labeled data
    where level 0 are assumed to be sample labels is also available in this module.

    The (row) index is interpreted as "features", often labels of spectral entities. Examples are
    m/z values, formulae or any format-specific labeling scheme. It may be hierarquical.
    """

    def __init__(self, df):
        self._validate(df)
        self._df = df

    @staticmethod
    def _validate(df):
        if not isinstance(df, pd.DataFrame):
            raise AttributeError("'ms' must be used with a Pandas DataFrame")

    @property
    def data_matrix(self):
        """The Pandas DataFrame holding the MS data, transposed to be usable as tidy"""
        return self._df.transpose(copy=True)

    @property
    def samples(self):
        """Get the different sample names."""
        if len(self._df.columns.names) > 1:
            return tuple(self._df.columns.levels[0])
        else:
            return tuple(self._df.columns)

    @samples.setter
    def samples(self, value):
        self._rebuild_col_level(value, 0)

    def _rebuild_col_level(self, value, level):
        cols = self._df.columns
        n = len(cols)
        metanames = cols.names
        nnames = len(metanames)
        # handle value
        if value is None or len(value) == 0:
            if level == 0:
                value = [f"Sample {i}" for i in range(1, n + 1)]
            else:
                value = [f"Info {i}" for i in range(1, n + 1)]
        elif _is_string(value):
            value = [value]
        else:
            value = list(value)
        nr = n // len(value)
        newstrs = []
        for s in value:
            newstrs.extend([s] * nr)
        if nnames == 1:
            self._df.columns = newstrs
            return
        cols = [list(c) for c in cols]
        for i, s in enumerate(newstrs):
            cols[i][level] = s
        newcols = [tuple(c) for c in cols]
        self._df.columns = pd.MultiIndex.from_tuples(newcols, names=metanames)

    @property
    def itersamples(self):
        if len(self._df.columns.names) > 1:
            self._df.columns = self._df.columns.remove_unused_levels()
            return self._df.columns.get_level_values(0)
        else:
            return tuple(self._df.columns)

    @property
    def feature_count(self):
        """Get the number of features."""
        return len(self._df.index)

    @property
    def sample_count(self):
        """Get the number of samples."""
        return len(self.samples)

    @property
    def no_labels(self):
        """True if there is only one (global) label 'no label'."""
        return True

    def info(self, all_data=False):
        if all_data:
            return dict(samples=self.sample_count, features=self.feature_count)
        s_table = {"sample": list(self.itersamples)}
        s_table["sample"].append((self.sample_count))
        indx_strs = [str(i) for i in range(self.sample_count)] + ["global"]
        return pd.DataFrame(s_table, index=indx_strs)

    def _get_subset_data(self, sample=None, info=None, no_drop_na=False):
        if sample is None and info is None:
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
                indexer.append(s)
            if len(indexer) == 1:
                indexer = indexer[0]
            # print('++++++++++++++++++')
            # print('indexer')
            # print(indexer)
            # print('++++++++++++++++++')
            df = self._df.loc[:, indexer]
            # print('++++++++++++++++++')
            # print('resulting df')
            # print(df)
            # print('++++++++++++++++++')
        if no_drop_na:
            df = df.copy()
        else:
            df = df.dropna(how="all")
        if isinstance(df, pd.DataFrame):
            if len(df.columns.names) > 1:
                df.columns = df.columns.remove_unused_levels()
        return df

    def take(self, **kwargs):
        return self._get_subset_data(**kwargs)

    def subset(self, **kwargs):
        return self.take(**kwargs)

    def features(self, **kwargs):
        df = self._get_subset_data(**kwargs)
        return df.index

    def pipe(self, func, drop_na=True, **kwargs):
        df = self._df
        df = df.pipe(func, **kwargs)
        if drop_na:
            df = df.dropna(how="all")
        if isinstance(df, pd.DataFrame):
            df.columns = df.columns.remove_unused_levels()
        return df

    def add_labels(self, labels=None, level_name="label"):
        newcols = create_multiindex_with_labels(
            self._df, labels=labels, level_name=level_name
        )
        self._df.columns = newcols
        return self._df.copy()
