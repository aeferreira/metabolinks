from collections import OrderedDict
import six

import pandas as pd
import numpy as np

from metabolinks import MSAccessor, UMSAccessor, demodata
from metabolinks.msaccessor import create_multiindex_with_labels
from metabolinks.utils import _is_string

def _ensure_data_frame(data):
    """Retrieves information from data structures building a dictionary.

       Accepts numpy array, pandas DataFrames or structures with a DataFrame member `data`."""

    # ensure data is a DataFrame, otherwise return numpy array as 'data' in info dict
    if not isinstance(data, pd.DataFrame):
        if hasattr(data, 'data_table') and isinstance(data.data_table, pd.DataFrame):
            data = data.data_table
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(np.array(data))
    return data

def gen_df(data, add_labels=None):
    """Ensure a Pandas DataFrame from data, create label level in columns if needed."""

    data = _ensure_data_frame(data)
    if add_labels is not None:
        data.columns = create_multiindex_with_labels(data, labels=add_labels)
    return data


def read_data_csv(filename, has_labels=False, sep='\t', **kwargs):
    if has_labels and 'header' not in kwargs:
        kwargs['header'] = [0, 1]
    df = pd.read_csv(filename, sep=sep, index_col=0, **kwargs)
    return gen_df(df)


def read_data_from_xcel(file_name, has_labels=False, verbose=True, **kwargs):

    datasets = OrderedDict()
    wb = pd.ExcelFile(file_name).book

    if verbose:
        print(f'------ Reading MS-Excel file - {file_name}')

    for sheetname in wb.sheet_names():
        if has_labels and 'header' not in kwargs:
            kwargs['header'] = [0, 1]

        # read data, first as a whole df. May have empty columns
        df = pd.read_excel(file_name, sheet_name=sheetname, **kwargs)
        all_columns = list(df.columns)
        d_columns = list(df.dropna(axis=1, how='all').columns)

        # find non-empty columns to split data in several tables
        # if necessary
        data_locs = []
        building = False
        for i, c in enumerate(all_columns):
            if c not in d_columns:
                building = False
                continue
            else:
                if not building:
                    data_locs.append([i])
                else:
                    data_locs[-1].append(i)
                building = True

        # now split in several tables if empty columns exist
        results = []
        for loc in data_locs:
            dataset = df.iloc[:, loc]
            dataset = dataset.dropna().set_index(dataset.columns[0])
            results.append(gen_df(dataset, has_labels))
            # print(dataset)
            # print('+++++++++++++++++++++++')
            # print(dataset.columns.names)
            # print(dataset.index.names)
            # print(dataset.columns)
            # print('*****************************')

        if verbose:
            print('\n- {} tables found in sheet "{}":'.format(len(results), sheetname))
            for table in results:
                size = len(table)
                print(table.ms.info())
                print(f'{size} features\n')
        datasets[sheetname] = results

    return datasets

# --------------------- MassTRIX search result files ---------

def read_MassTRIX(fname, unfolded=False):
    """Reads a MassTRIX file into a Pandas DataFrame object.
       
       On the process, the last line is moved to the beginning and
       is read as the header."""
    
    # store lines in a list
    with open(fname) as f:
        lines = [line.strip() for line in f]
    
    if not unfolded:
        # move the last line to the beginning
        moved_list = [lines[-1]]
        moved_list.extend(lines[:-1]) # last line is not included
        lines = moved_list

    # read from the list of strings in memory
    return pd.read_csv(six.StringIO('\n'.join(lines)), sep='\t')

# -----------------------------------------------------------

if __name__ == '__main__':
    print('test construction from numpy array')
    n = 12
    nrows = 4
    data = np.array(range(12 * 4)).reshape(nrows, n)
    print(f'data = \n{data}')
    df = gen_df(data)
    print(df)
    print(df.columns.names)
    print(df.index.names)
    print(df.columns.levels)
    print('-------------------------')
    print('test construction from minimal pandas dataFrame')
    exp_data = pd.DataFrame(
        data,
        index=['exp1', 'exp2', 'exp3', 'exp4'],
        columns=[f'SE{i}' for i in range(1, 13)],
    )
    print(f'data = \n{exp_data}')
    df = gen_df(exp_data)
    print(df)
    print(df.columns.names)
    print(df.index.names)
    print(df.columns.levels)
    print('-------------------------')
    print('test construction from explicit data')
    features = 'F1 F2 F3 F4'.split()
    # labels = None
    labels = ['L1', 'L2']
    samples = 'S1 S2 S3 S4'.split()
    info_types = ['I', 'RT', 'delta']
    print(f'data = \n{data}')
    print(f'labels = {labels}')
    print(f'features = {features}')
    print(f'samples = {samples}')
    print(f'info_types = {info_types}')
    df = gen_df(
        data,
        labels=labels,
        features=features,
        samples=samples,
        info_types=info_types,
        features_name='m/z',
    )
    print(df)
    print(df.columns.names)
    print(df.index.names)
    print(df.columns.levels)
    print('-------------------------')
    print('test construction from explicit data')
    samples = [f'S{i}' for i in range(1, 13)]
    info_types = ['I']
    print(f'data = \n{data}')
    print(f'features = {features}')
    print(f'labels = {labels}')
    print(f'samples = {samples}')
    print(f'info_types = {info_types}')
    df = gen_df(
        data,
        labels=labels,
        features=features,
        samples=samples,
        info_types=info_types,
        features_name='m/z',
    )
    print(df)
    print(df.columns.names)
    print(df.index.names)
    print(df.columns.levels)
    print('-------------------------')
    print('test construction from explicit data')
    samples = None
    info_types = None
    print(f'data = \n{data}')
    print(f'features = {features}')
    print(f'labels = {labels}')
    print(f'samples = {samples}')
    print(f'info_types = {info_types}')
    df = gen_df(
        data, labels=labels, features=features, samples=samples, info_types=info_types
    )
    print(df)
    print(df.columns.names)
    print(df.index.names)
    print(df.columns.levels)
    print('-------------------------')
    print('test construction from already generated pandas DataFrame and new data')
    labels = None
    features = None
    samples = 'S1 S2 S3 S4'.split()
    info_types = ['I', 'RT', 'delta']
    print(f'data = \n{df}')
    print(f'features = {features}')
    print(f'labels = {labels}')
    print(f'samples = {samples}')
    print(f'info_types = {info_types}')
    df = gen_df(df, samples=samples, info_types=info_types, info_name='data')
    print(df)
    print(df.columns.names)
    print(df.index.names)
    print(df.columns.levels)

    print('MSDataSet from string data (as io stream) ------------\n')
    dataset = read_data_csv(six.StringIO(demodata.demo_data1()))
    print(dataset)
    print('-- info --------------')
    print(dataset.ms.info())
    print('-- global info---------')
    print(dataset.ms.info(all_data=True))
    print('-----------------------')

    print('MSDataSet from string data (as io stream) ------------\n')
    dataset = read_data_csv(six.StringIO(demodata.demo_data2()), has_labels=True)
    print(dataset)
    print('-- info --------------')
    print(dataset.ms.info())
    print('-- global info---------')
    print(dataset.ms.info(all_data=True))
    print('-----------------------')

    # Reading from Excel ----------
    file_name = 'sample_data.xlsx'
    import os

    _THIS_DIR, _ = os.path.split(os.path.abspath(__file__))
    fname = os.path.join(_THIS_DIR, 'data', file_name)
    data_sets = read_data_from_xcel(fname, header=[0, 1], verbose=True)

    for d in data_sets:
        print(d)
        print('+++++++++++++++++++++++++++++')
        for t in data_sets[d]:
            print('\n----- table -----------')
            # print(t.columns.names)
            # print(t.index.names)
            # print(t.columns.levels)
            print(t)
        print('#############################')


# def default_header_csv(self, s, sep=None, with_labels=False):
#     # this returns a header suitable for various metabolomics tools
#     if sep is None:
#         sep = '\t'
#     lines = []
#     line = ['Sample'] + s.sample_names
#     line = sep.join(['"{}"'.format(n) for n in line])
#     lines.append(line)
#     if with_labels and s.labels is not None:
#         line = ['Label'] + s.labels
#         line = sep.join(['"{}"'.format(n) for n in line])
#         lines.append(line)
#     return '\n'.join(lines)

# def common_label_mz(self, label1, label2):
#     mz1 = self.label(label1).mz
#     mz2 = self.label(label2).mz
#     u = np.intersect1d(mz1, mz2)
#     return u

# def exclusive_label_mz(self, label):
#     # build list of unique other labels
#     slabels = [lbl for lbl in self.unique_labels() if lbl != label]

#     remaining = self.label(label).mz
#     for lbl in slabels:
#         remaining = np.setdiff1d(remaining, self.label(lbl).mz)
#     return remaining

# @property
# def exclusive_mz(self):
#     res = OrderedDict()
#     for label in self.labels:
#         slabels = [lbl for lbl in self.unique_labels() if lbl != label]

#         remaining = self.label(label).mz
#         for lbl in slabels:
#             remaining = np.setdiff1d(remaining, self.label(lbl).mz)
#         res[label] = remaining
#     return res

# print('\nFiltered fewer than 2 per label')
# print('\n-- Original\n')
# print(spectra)
# print('\n-- With a minimum of 2 replicates\n')
# print(spectra.rep_at_least(minimum=2))

# print('\nUnfolding spectra ----------')
# unfolded = spectra.unfold()

# for u in unfolded:
#     print(u)
#     print('+++++')

# print('\nSaving aligned spectra into a file ----------')
# samplefile = StringIO()
# spectra.to_csv(samplefile, sep=',', with_labels=True)
# print('\n--- Resulting file:')
# print(samplefile.getvalue())

# print('\n--- Reading back:')
# samplefile.seek(0)
# spectra2 = read_aligned_spectra(samplefile, labels=True, sep=',')
# print(spectra2,'\n')
# spectra2.data.info()
# print(spectra2.info())

# print('\nCommon m/z between labels (v1,v2) ----------')
# print(spectra.common_label_mz('v1', 'v2'))

# print('\nm/z values exclusive to label v1 ----------')
# print(spectra.exclusive_label_mz('v1'))

# print('\nm/z values exclusive to each label ----------')
# for label, values in spectra.exclusive_mz.items():
#     print('label:', label)
#     print(values)

# print('\nComputing similarity measures ----------')

# print(mz_similarity(spectra))
