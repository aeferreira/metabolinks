from __future__ import print_function

import pandas as pd

"""Utility functions for CSV i/o and simple transformations of spectra.

Spectra are implemented as Pandas DataFrames, with columns[0] == 'm/z' and
the remaining columns are sample names."""

class Spectrum(object):
    def __init__(self, df=None):
        self._df = None
        if df is not None:
            self._df = df.copy()
    
    def get_df(self):
        return self._df

    def to_csv(self, filename, mz_name=None, sep=None, **kwargs):
        header = True
        if mz_name is not None:
            header = [mz_name] + list(self._df.columns[1:])
        if sep is None:
            sep = '\t'
        self._df.to_csv(filename, header=header, sep=sep, 
                        index=False, **kwargs)

class AlignedSpectra(object):
    def __init__(self, df=None):
        self._df = None
        if df is not None:
            self._df = df.copy()
    
    def get_df(self):
        return self._df

    def to_csv(self, filename, mz_name=None, sep=None, **kwargs):
        header = True
        if mz_name is not None:
            header = [mz_name] + list(self._df.columns[1:])
        if sep is None:
            sep = '\t'
        self._df.to_csv(filename, header=header, sep=sep, 
                        index=False, **kwargs)

def read_spectrum(filename):
    s = pd.read_table(filename, index_col=False)
    # keep only the first two columns
    s = s.iloc[:, [0,1]]
    # force 'm/z' name for first column
    newnames = ['m/z', s.columns[1]]
    s.columns = newnames
    return Spectrum(s)


def read_aligned_spectra(filename):
    s = pd.read_table(filename, index_col=False)
    # force 'm/z' name for first column
    newnames = ['m/z'] + list(s.columns[1:])
    s.columns = newnames
    return AlignedSpectra(s)


if __name__ == '__main__':

    fname = '../example_data/aligned_spectra.txt'
    sname = '../example_data/aligned_spectra_test.txt'
    
    print('Reading spectrum', fname, '----------')
    spectrum = read_spectrum(fname)
    spectrum.get_df().info()
    print(spectrum.get_df().head(10))
    print('Reading aligned spectra', fname, '----------')
    spectra = read_aligned_spectra(fname)
    spectra.get_df().info()
    print(spectra.get_df().head(15))
    print('Saving aligned spectra into', sname, '----------')
    spectra.to_csv(sname, mz_name='Samples')

