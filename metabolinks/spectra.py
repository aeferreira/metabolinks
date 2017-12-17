from __future__ import print_function

import numpy as np
import pandas as pd

"""Classes representing MS spectra.

Spectra data are implemented as Pandas DataFrames, with columns[0] == 'm/z'.
I/O to CSV (TSV) and simple transformations are supported."""

class Spectrum(object):
    def __init__(self, df=None, sample_name=None, label=None):
        self._df = None
        self.sample_similarity = None
        if df is not None:
            self._df = df.copy()
        self.sample_name = None
        self.label = None
        if sample_name is not None:
            self.sample_name = sample_name
        else:
            if self._df is not None:
                self.sample_name = self._df.columns[1]
        if label is not None:
            self.label = label
    
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
    def __init__(self, df=None, sample_names=None, labels=None):
        self._df = None
        if df is not None:
            self._df = df.copy()
        self.labels = None
        if labels is not None:
            self.labels = labels
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

    def to_csv(self, filename, header_func=None, sep=None, 
               no_report_columns=True, **kwargs):
        if sep is None:
            sep = '\t'
        out_df = self._df.copy()
        if no_report_columns:
            if '#samples' in self._df.columns:
                s_loc = self._df.columns.get_loc('#samples')
                out_df = self._df.iloc[:, :s_loc]
        if header_func is None:
            out_df.to_csv(filename, header=True, sep=sep, index=False, **kwargs)
        else:
            with open(filename, 'w') as of:
                header = header_func(self)
                of.write(header+'\n')
                out_df.to_csv(of, header=False, sep=sep, 
                              index=False, **kwargs)
                

    def fillna(self, value):
        vdict = dict([(n, value) for n in self.sample_names])
        self._df.fillna(vdict, inplace=True)
        
    def unfold(self):
        res = []
        for i, name in enumerate(self.sample_names):
            df = self._df.iloc[:, [0, i+1]].dropna()
            res.append(Spectrum(df, sample_name=name))
        return res
    
    def compute_similarity_measures(self):
        # compute counts and Jaccard index by samples
        self.sample_similarity = None
        spectra = self.unfold()
        
        n = len(spectra)
        self.sample_similarity = None
        smatrix = np.zeros((n, n))
        for i1 in range(n-1):
            for i2 in range(i1+1, n):
                s1 = spectra[i1]
                s2 = spectra[i2]
                mz1 = s1.data['m/z']
                mz2 = s2.data['m/z']
                print(i1, s1.sample_name, len(mz1))
                print(i2, s2.sample_name, len(mz2))
                smatrix[i1, i1] = len(mz1)
                smatrix[i2, i2] = len(mz2)
                set1 = set(mz1.values)
                set2 = set(mz2.values)
                u12 = set1.union(set2)
                i12 = set1.intersection(set2)
                smatrix[i1, i2] = len(i12)
                jaccard = len(i12) / len(u12)
                smatrix[i2, i1] = jaccard
        self.sample_similarity = smatrix
                
                
        

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
    def header(s):
        return'Samples\t'+'\t'.join(s.sample_names)
    spectra.to_csv(sname, header_func=header)

    print('\nUnfolding spectra ----------')
    spectra = read_aligned_spectra(fname)
    unfolded = spectra.unfold()
    
    for u in unfolded:
        print('Spectrum', u.sample_name)
        print(u.data.head(15))
        print('+++++')
    
    print('\nComputing similarity ----------')
    
    spectra.compute_similarity_measures()
    
    print(spectra.sample_similarity)
        
