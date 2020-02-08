from collections import OrderedDict

import pandas as pd
import numpy as np

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
    info['features_name'] = data.index.names[0] # this may be None
    
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
    #print('###############################\noriginal info')
    info = extract_info_from_ds(data)
    #print(info_dict)
    
    # overwrite using function arguments
    # function arguments to consider...
    arg_names = ('features', 'samples', 'labels', 'info_types', 'features_name', 'info_name')
    # A None value for the following is meaningful: overwrite. If others are None, ignore them...
    arg_names_nonOK = ('features_name', 'info_name')
    overwrite_dict = {a: kwargs[a] for a in arg_names if a in kwargs}
    overwrite_dict = {a: v for (a, v) in overwrite_dict.items() if (a in arg_names_nonOK) or (v is not None)}
    #print('kwargs info\n', overwrite_dict)
    info.update(overwrite_dict)
    #print('updated info\n',info_dict)
    #print('###############################')
    
    # get data values
    data = info['data'] # data must always exist
    nrows, n = data.shape
    
    # handle labels
    labels = info.setdefault('labels', ['no label'])
    nr = n // len(labels)
    alabels = []
    for s in labels:
        alabels.extend([s]*nr)
    
    # handle samples
    samples = info.setdefault('samples', [f'Sample {i}' for i in range(1, n + 1)])
    nr = n // len(samples)
    asamples = []
    for s in samples:
        asamples.extend([s]*nr)
    
    # handle info types and build column index from (labels, samples, info_types)
    if 'info_types' not in info:
        ci =  pd.MultiIndex.from_arrays((alabels, asamples), names=['label', 'sample'])
    else:
        atypes = info['info_types'] * len(samples)
        info_name = info.setdefault('info_name', '')
        if info_name is None:
            info_name = ''
        ci = pd.MultiIndex.from_arrays((alabels, asamples, atypes),
                                       names=['label', 'sample', info_name])
    
    # handle features (row index)
    features = info.setdefault('features', [f'F{i}' for i in range(nrows)])
    features_name = info.setdefault('features_name', None)
    fi = pd.Index(features, name=features_name)

    # build pandas DataFrame
    return pd.DataFrame(data, index=fi, columns=ci)


    # def to_csv(self, filename, header_func=None, sep=None, with_labels=False,
    #            no_meta_columns=True, **kwargs):
    #     if sep is None:
    #         sep = '\t'
    #     out_df = self._df.copy()
    #     if no_meta_columns:
    #         out_df = out_df.iloc[:, :len(self.sample_names)]
    #     if header_func is None:
    #         header_func = self.default_header_csv
    #     # prepend output with result of header_func
    #     needs_to_close = False
    #     if _is_string(filename):
    #         of = open(filename, 'w') 
    #         needs_to_close = True
    #     else:
    #         of = filename

    #     header = header_func(self, sep=sep, with_labels=with_labels) + '\n'
    #     of.write(header)
    #     out_df.to_csv(of, header=False, index=True, sep=sep, **kwargs)

    #     if needs_to_close:
    #         of.close()


def read_spectrum(filename, label=None):
    s = pd.read_table(filename, index_col=False)
    # keep only the first two columns
    s = s.iloc[:, [0,1]]
    s = s.set_index(s.columns[0])
    return Spectrum(s, label=label)


def read_aligned_spectra(filename, labels=None, **kwargs):
    if labels == True:
        s = pd.read_table(filename, header=[0,1], **kwargs)
    else:
        s = pd.read_table(filename, index_col=False, **kwargs)
    
    if labels == True:
        snames = s.columns.get_level_values(0)
        lnames = s.columns.get_level_values(1)[1:]
        s.columns = snames
        labels = lnames

    s = s.set_index(s.columns[0])
    s.index.name = 'features'
    return MSDataSet(s, labels=labels)


def read_spectra_from_xcel(file_name,
                           sample_names=None,
                           labels=None,
                           header_row=1,
                           common_mz= False,
                           verbose=True):

    spectra_table = OrderedDict()

    wb = pd.ExcelFile(file_name).book
    header = header_row - 1

    if verbose:
        print ('------ Reading MS-Excel file - {}'.format(file_name))

    for sheetname in wb.sheet_names():

        # if sample_names argument if present (not None) then
        # if an integer, read row with sample names,
        # otherwise sample_names must a list of names for samples
        snames = []
        if sample_names is not None:
            if isinstance(sample_names, six.integer_types):
                sh = wb.sheet_by_name(sheetname)
                snames = sh.row_values(sample_names - 1)
                snames = [s for s in snames if len(s.strip()) > 0]
            else:
                snames = sample_names

        # read data (and discard empty xl columns)
        df = pd.read_excel(file_name,
                           sheet_name=sheetname,
                           header=header)
        df = df.dropna(axis=1, how='all')

        # if sample names are not set then
        # use "2nd columns" headers as sample names
        # if common_mz then use headers from position 1
        if len(snames) > 0:
            pass # already fetched
        else:
            if common_mz:
                snames = df.columns[1:]
            else:
                snames = df.columns[1::2]

        # split in groups of two (each group is a spectrum)
        results = []
        if not common_mz:
            j = 0
            for i in range(0, len(df.columns), 2):
                spectrum = df.iloc[:, i: i+2]
                spectrum.index = range(len(spectrum))
                spectrum = spectrum.dropna()
                spectrum = spectrum.set_index(spectrum.columns[0])
                spectrum = Spectrum(spectrum, sample_name=snames[j])
                results.append(spectrum)

                j = j + 1
        else:
            j = 0
            for i in range(1, len(df.columns)):
                spectrum = df.iloc[:, [0, i]]
                spectrum.index = range(len(spectrum))
                spectrum = spectrum.dropna()
                spectrum = spectrum.set_index(spectrum.columns[0])
                spectrum = Spectrum(spectrum, sample_name=snames[j])
                results.append(spectrum)

                j = j + 1


        if labels is not None:
            if _is_string(labels):
                labels = [labels] * len(results)
            for lbl, spectrum in zip(labels, results):
                spectrum.set_label(lbl)

        if verbose:
            print ('- {} spectra found in sheet "{}":'.format(len(results), sheetname))
            for spectrum in results:
                name = spectrum.sample_name
                label = spectrum.label
                size = len(spectrum)
                print ('{:5d} peaks in sample {}, with label {}'.format(size, name, label))
        spectra_table[sheetname] = results

    return spectra_table


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
    exp_data = pd.DataFrame(data, index=['exp1', 'exp2', 'exp3', 'exp4'], columns=[f'SE{i}' for i in range(1,13)])
    print(f'data = \n{exp_data}')
    df = gen_df(exp_data)
    print(df)
    print(df.columns.names)
    print(df.index.names)
    print(df.columns.levels)
    print('-------------------------')
    print('test construction from explicit data')
    features = 'F1 F2 F3 F4'.split()
    #labels = None
    labels = ['L1', 'L2']
    samples = 'S1 S2 S3 S4'.split()
    info_types = ['I', 'RT', 'delta']
    print(f'data = \n{data}')
    print(f'labels = {labels}')
    print(f'features = {features}')
    print(f'samples = {samples}')
    print(f'info_types = {info_types}')
    df = gen_df(data, labels=labels, features=features, samples=samples, info_types=info_types, features_name='m/z')
    print(df)
    print(df.columns.names)
    print(df.index.names)
    print(df.columns.levels)
    print('-------------------------')
    print('test construction from explicit data')
    samples = [f'S{i}' for i in range(1,13)]
    info_types = ['I']
    print(f'data = \n{data}')
    print(f'features = {features}')
    print(f'labels = {labels}')
    print(f'samples = {samples}')
    print(f'info_types = {info_types}')
    df = gen_df(data, labels=labels, features=features, samples=samples, info_types=info_types, features_name='m/z')
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
    df = gen_df(data, labels=labels, features=features, samples=samples, info_types=info_types)
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

##     print('\nReading from Excel ----------')
##     file_name='data_to_align.xlsx'
##     spectra3 = read_spectra_from_xcel(file_name,
##                            sample_names=1, header_row=1)

