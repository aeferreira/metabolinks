from __future__ import print_function

import numpy as np
import pandas as pd

"""Classes representing MS peak lists.

Peak data are implemented as Pandas DataFrames, with columns[0] == 'm/z'.
I/O to CSV (TSV) and simple transformations are supported."""

class Spectrum(object):
    def __init__(self, df=None, sample_name=None, label=None):
        self._df = None
        if df is not None:
            self._df = df.copy()
        self.sample_name = None
        self.label = label
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
    def __init__(self, df=None, sample_names=None, labels=None):
        self._df = None
        if df is not None:
            self._df = df.copy()
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
        self.sample_similarity = None
        self.label_similarity = None
        self.unique_labels = None

    
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
            if self.labels is None:
                label = None
            else:
                label = self.labels[i]
            res.append(Spectrum(df, sample_name=name, label=label))
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
        
        if self.labels is not None:
            self.label_similarity = None
            self.unique_labels = None
            # build list of unique labels
            slabels = [self.labels[0]]
            for i in range(1, len(self.labels)):
                label = self.labels[i]
                if label not in slabels:
                    slabels.append(label)
            # build dict of lists grouping samples with a given label
            groups = {}
            for label in slabels:
                groups[label] = [s for s in spectra if s.label == label]
            # find union of m/z values for each label (as a set)
            mzs = {}
            for label in slabels:
                # print('Label', label, len(groups[label]), 'spectra')
                gl = groups[label]
                mz = set(gl[0].data['m/z'])
                #print(0)
                #print(len(mz), mz)
                for i in range(1, len(gl)):
                    mz = mz.union(set(gl[i].data['m/z']))
                    #print(i)
                    #print(len(mz), mz)
                mzs[label] = mz
            # compute intersection counts and Jaccard index
            n = len(slabels)
            lmatrix = np.zeros((n, n))
            for i1 in range(n-1):
                for i2 in range(i1+1, n):
                    label1 = slabels[i1]
                    label2 = slabels[i2]
                    set1 = mzs[label1]
                    set2 = mzs[label2]
                    lmatrix[i1, i1] = len(set1)
                    lmatrix[i2, i2] = len(set2)
                    u12 = set1.union(set2)
                    i12 = set1.intersection(set2)
                    lmatrix[i1, i2] = len(i12)
                    jaccard = len(i12) / len(u12)
                    lmatrix[i2, i1] = jaccard
            self.label_similarity = lmatrix
            self.unique_labels = slabels



def read_spectrum(filename, label=None):
    s = pd.read_table(filename, index_col=False)
    # keep only the first two columns
    s = s.iloc[:, [0,1]]
    # force 'm/z' name for first column
    newnames = ['m/z', s.columns[1]]
    s.columns = newnames
    return Spectrum(s, label=label)


def read_aligned_spectra(filename, labels=None):
    s = pd.read_table(filename, index_col=False)
    # force 'm/z' name for first column
    newnames = ['m/z'] + list(s.columns[1:])
    s.columns = newnames
    return AlignedSpectra(s, labels=labels)


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

    print('\nReading aligned spectra', fname, '----------')
    spectra = read_aligned_spectra(fname)
    print('Sample names:', spectra.sample_names)
    spectra.data.info()
    print(spectra.data)

    print('\nFilling with zeros ----------')
    spectra.fillna(0)

    print('Saving aligned spectra into', sname, '----------')
    def header(s):
        return'Samples\t'+'\t'.join(s.sample_names)
    spectra.to_csv(sname, header_func=header)

    print('\nUnfolding spectra ----------')
    labels=['v1', 'v1', 'v1', 'v2', 'v2', 'v2']
    spectra = read_aligned_spectra(fname, labels=labels)
    unfolded = spectra.unfold()
    
    for u in unfolded:
        print('Spectrum', u.sample_name, u.label, 'size:', len(u.data))
        print(u.data)
        print('+++++')
    
    print('\nComputing similarity ----------')
    
    spectra.compute_similarity_measures()
    
    print('\n- Sample similarity --')
    print(spectra.sample_similarity)
        
    print('\n- Label similarity --')
    print(spectra.label_similarity)
