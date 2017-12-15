from __future__ import print_function

import pandas as pd

"""Classes representing MS spectra.

Spectra data are implemented as Pandas DataFrames, with columns[0] == 'm/z'.
I/O to CSV (TSV) and simple transformations are supported."""

class Spectrum(object):
    def __init__(self, df=None, sample_name=None):
        self._df = None
        if df is not None:
            self._df = df.copy()
        self.sample_name = None
        if sample_name is not None:
            self.sample_name = sample_name
        else:
            if self._df is not None:
                self.sample_name = self._df.columns[1]
    
    @property
    def data(self):
        return self._df

    def to_csv(self, filename, mz_name=None, sep=None, **kwargs):
        header = True
        if mz_name is not None:
            header = [mz_name] + list(self._df.columns[1:])
        if sep is None:
            sep = '\t'
        self._df.to_csv(filename, header=header, sep=sep, 
                        index=False, **kwargs)
    
    def mz_to_csv(self, filename, mz_name=None, **kwargs):
        outdf = self._df.iloc[:,0:1]
        if mz_name is not None:
            outdf.columns = [mz_name]
        outdf.to_csv(filename, header=mz_name, index=False, **kwargs)

class AlignedSpectra(object):
    def __init__(self, df=None, sample_names=None):
        self._df = None
        if df is not None:
            self._df = df.copy()
        self.sample_names = None
        if sample_names is not None:
            self.sample_names = sample_names
        else:
            if self._df is not None:
                if '#samples' in self._df.columns:
                    s_loc = self._df.columns.get_loc('#samples')
                    self.sample_names = list(self._df.columns[1:s_loc])
                else:
                    self.sample_names = list(self._df.columns[1:])
    
    @property
    def data(self):
        return self._df

    def to_csv(self, filename, mz_name=None, sep=None, 
               no_report_columns=True, **kwargs):
        if sep is None:
            sep = '\t'
        out_df = self._df.copy()
        header = list(self._df.columns)
        if mz_name is not None:
            header = [mz_name] + list(self._df.columns[1:])
        out_df.columns = header
        if no_report_columns:
            if '#samples' in self._df.columns:
                s_loc = self._df.columns.get_loc('#samples')
                out_df = self._df.iloc[:, :s_loc]
            
        out_df.to_csv(filename, header=True, sep=sep, 
                        index=False, **kwargs)

    def fillna(self, value):
        vdict = dict([(n, value) for n in self.sample_names])
        self._df.fillna(vdict, inplace=True)
        

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
    mzname = '../example_data/spectrum_mz.txt'
    
    print('Reading spectrum', fname, '----------')
    spectrum = read_spectrum(fname)
    print('Sample name:', spectrum.sample_name)
    spectrum.data.info()
    print(spectrum.data.head(10))

    print('Saving spectrum mz into', mzname, '----------')
    spectrum.mz_to_csv(mzname, mz_name='moverz')

    print('Reading aligned spectra', fname, '----------')
    spectra = read_aligned_spectra(fname)
    print('Sample names:', spectra.sample_names)
    spectra.data.info()
    print(spectra.data.head(15))

    print('Filling with zeros ----------')
    spectra.fillna(0)

    print('Saving aligned spectra into', sname, '----------')
    spectra.to_csv(sname, mz_name='Samples')

