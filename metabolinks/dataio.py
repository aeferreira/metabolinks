from collections import OrderedDict
import six

import pandas as pd
import numpy as np

from metabolinks import MSAccessor, demodata
from metabolinks.utils import _is_string


def extract_info_from_ds(data):
    """Retrieves information from data structures building a dictionary.

       Accepts numpy array, pandas DataFrames or structures with a DataFrame member `data`.
    """

    info = {}

    # ensure data is a DataFrame, otherwise return numpy array as 'data' in info dict
    if not isinstance(data, pd.DataFrame):
        if hasattr(data, 'data_table') and isinstance(data.data_table, pd.DataFrame):
            data = data.data_table
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if not isinstance(data, pd.DataFrame):
        info['data'] = np.array(data)
        return info

    # data values
    info['data'] = np.array(data)

    # features and features_name
    info['features'] = data.index.values
    info['features_name'] = data.index.names[0]  # this may be None

    # labels, samples and information types
    ncl = len(data.columns.names)
    if ncl in [2, 3]:
        info['labels'] = data.columns.get_level_values(0)
        info['samples'] = data.columns.get_level_values(1)
    if ncl == 1:
        info['samples'] = data.columns
    if ncl == 3:
        info['info_types'] = data.columns.levels[2]
        info['info_name'] = data.columns.names[2]
    return info


def gen_df(data, **kwargs):
    """Generate a pandas DataFrame with appropriate index and columns from data and function arguments.

       Information is first retrieved from `data`. Next, arguments overwrite information
       and, lastly, default values are provided.
       `data` can be a numpy array, a pandas DataFrames or structures with a DataFrame member `data`.
    """

    # "Parse" data to retrieve information
    info = extract_info_from_ds(data)

    # overwrite using function arguments
    # function arguments to consider...
    arg_names = (
        'features',
        'samples',
        'labels',
        'info_types',
        'features_name',
        'info_name',
    )
    # A None value for the following is meaningful: overwrite.
    # if others are None, ignore them...
    arg_names_nonOK = ('features_name', 'info_name')
    overwrite_dict = {a: kwargs[a] for a in arg_names if a in kwargs}
    overwrite_dict = {
        a: v
        for (a, v) in overwrite_dict.items()
        if (a in arg_names_nonOK) or (v is not None)
    }
    info.update(overwrite_dict)
    # print('updated info\n',info)
    # print('###############################')

    # get data values
    data = info['data']  # data must always exist
    nrows, n = data.shape

    # handle labels
    labels = info.setdefault('labels', ['no label'])
    nr = n // len(labels)
    alabels = []
    for s in labels:
        alabels.extend([s] * nr)

    # handle samples
    samples = info.setdefault('samples', [f'Sample {i}' for i in range(1, n + 1)])
    nr = n // len(samples)
    asamples = []
    for s in samples:
        asamples.extend([s] * nr)
    # print('asamples\n',asamples)
    # print('###############################')

    # handle info types and build column index from (labels, samples, info_types)
    if 'info_types' not in info:
        ci = pd.MultiIndex.from_arrays((alabels, asamples), names=['label', 'sample'])
    else:
        atypes = info['info_types'] * len(samples)
        info_name = info.setdefault('info_name', '')
        if info_name is None:
            info_name = ''
        ci = pd.MultiIndex.from_arrays(
            (alabels, asamples, atypes), names=['label', 'sample', info_name]
        )

    # handle features (row index)
    features = info.setdefault('features', [f'F{i}' for i in range(nrows)])
    features_name = info.setdefault('features_name', None)
    fi = pd.Index(features, name=features_name)

    # build pandas DataFrame
    return pd.DataFrame(data, index=fi, columns=ci)


def read_data_csv(filename, has_labels=False, sep='\t', **kwargs):
    if has_labels:
        df = pd.read_csv(filename, header=[0, 1], sep=sep, index_col=0, **kwargs)
    else:
        df = pd.read_csv(filename, sep=sep, index_col=0, **kwargs)
    # print('*****************************')
    # print(df)
    # print(df.columns.names)
    # print(df.index.names)
    # print(df.columns)
    # print('*****************************')
    return gen_df(df)


def read_data_from_xcel(
    file_name,
    sample_row=None,
    labels=None,
    header_row=1,
    common_features=False,
    verbose=True,
):

    datasets = OrderedDict()

    wb = pd.ExcelFile(file_name).book
    header = header_row -1

    if verbose:
        print(f'------ Reading MS-Excel file - {file_name}')

    for sheetname in wb.sheet_names():

        # if sample_row argument is present (not None) then, read row with sample names.
        # for setting new sample names, use .ms.set_samples(new_names) on the values of the result
        snames = []
        if sample_row is not None:  #must be an integer
            sh = wb.sheet_by_name(sheetname)
            snames = sh.row_values(sample_row - 1)
            snames = [s for s in snames if len(s.strip()) > 0]

        # read data (and discard empty xl columns)
        df = pd.read_excel(file_name, sheet_name=sheetname, header=header)
        df = df.dropna(axis=1, how='all')
        print('+++++++++++++++++++++++')
        print(df)
        print('+++++++++++++++++++++++')

        # if sample names are not set then
        # use "2nd columns" headers as sample names
        # if common_features then use headers from position 1
        if len(snames) > 0:
            pass  # already fetched
        else:
            if common_features:
                snames = df.columns[1:]
            else:
                snames = df.columns[1::2]

        # split in groups of two (each group is a different table)
        results = []
        if not common_features:
            j = 0
            for i in range(0, len(df.columns), 2):
                spectrum = df.iloc[:, i : i + 2]
                spectrum.index = range(len(spectrum))
                spectrum = spectrum.dropna().set_index(spectrum.columns[0])
                results.append(gen_df(spectrum, samples=[snames[j]]))
                j = j + 1
        else:
            j = 0
            for i in range(1, len(df.columns)):
                spectrum = df.iloc[:, [0, i]]
                spectrum.index = range(len(spectrum))
                spectrum = spectrum.dropna().set_index(spectrum.columns[0])
                results.append(gen_df(spectrum, samples=[snames[j]]))
                j = j + 1

        if verbose:
            print('\n- {} tables found in sheet "{}":'.format(len(results), sheetname))
            for table in results:
                size = len(table)
                print(table.ms.info())
                print(f'{size} features\n')
        datasets[sheetname] = results

    return datasets


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
    file_name='sample_data.xlsx'
    import os

    _THIS_DIR, _ = os.path.split(os.path.abspath(__file__))
    fname = os.path.join(_THIS_DIR, 'data', file_name)
    data_sets = read_data_from_xcel(fname, sample_row=1, header_row=1, verbose=True)

    for d in data_sets:
        print(d)
        print('-----------------------------')
        for t in data_sets[d]:
            print('\n----- table')
            print(t.columns.names)
            print(t.index.names)
            print(t.columns.levels)
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


# print('\nSaving aligned spectra into a file ----------')
# samplefile = StringIO()
# spectra.to_csv(samplefile, sep=',')
# print('\n--- Resulting file:')
# print(samplefile.getvalue())

# print('\n--- Reading back:')
# samplefile.seek(0)
# spectra2 = read_aligned_spectra(samplefile, sep=',')
# print(spectra2,'\n')
# spectra2.data.info()
# print(spectra2.info())

# print('\nReading aligned spectra (all of them) ----------')
# labels=['v1', 'v1', 'v1', 'v2', 'v2', 'v2', 'v3', 'v3'] # last 2 exceed
# sample.seek(0)
# spectra = read_aligned_spectra(sample, labels=labels)
# print(spectra,'\n')
# spectra.data.info()
# print(spectra.info())

# print('\nFiltered fewer than 2 per label')
# print('\n-- Original\n')
# print(spectra)
# print('\n-- With a minimum of 2 replicates\n')
# print(spectra.rep_at_least(minimum=2))

# print('\nSpectra with missing values filled with zeros ----------')
# spectrazero = spectra.fillna(0)
# print(spectrazero.data)

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
