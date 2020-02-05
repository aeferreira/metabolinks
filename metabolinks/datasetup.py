import pandas as pd
import numpy as np

def extract_info_from_ds(data):
    """Retrieves information from data structures building a dictionary.

       Accepts numpy array, pandas DataFrames or structures with a DataFrame member `data`.
    """

    info = {}
    
    # ensure data is a DataFrame, otherwise return numpy array as 'data' in info dict
    if not isinstance(data, pd.DataFrame):
        if hasattr(data, 'data') and isinstance(data.data, pd.DataFrame):
            data = data.data
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
        info['labels'] = data.columns.levels[0]
        info['samples'] = data.columns.levels[1]
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
