"""Classes representing MS peak lists.

There are two types of peak lists:

- single (Spectrum)
- aligned (AlignedSpectra). AlignedSpectra represents
multiple peak lists that share the same m/z values.

Simple data handling functions are implemented as well as I/O to CSV (TSV).

Much more data handling procedures are available through the property `data`
which is a Pandas DataFrame.

"""

from __future__ import print_function

from collections import OrderedDict
import six

import numpy as np
import pandas as pd

from metabolinks.utils import _is_string
from metabolinks.similarity import compute_similarity_measures

class Spectrum(object):
    """A single peak list.

    The underlying data (property `data`) is implemented in a Pandas DataFrame.
    The index holds m/z values or any other peak labels.
    There should be only one column, which holds (ususally) intensity data.

    Attributes
    ----------
    sample_name : str
        The name of this sample.
    label : str
        Optional sample label, used in classification tasks.

    """

    def __init__(self, df=None, sample_name=None, label=None):
        if df is not None:
            self._df = df.copy()
        else:
            self._df = df
        
        self.label = label
        
        self.sample_name = None
        if sample_name is not None:
            self.sample_name = sample_name
        else:
            # read sample name from the name of the only column
            if self._df is not None:
                self.sample_name = self._df.columns[0]
    
    @property
    def data(self):
        """The Pandas DataFrame holding the MS data."""
        return self._df

    def __str__(self):
        res = ['Sample: ' + self.sample_name]
        res.append('Label: {}'.format(self.label))
        res.append('{} peaks'.format(len(self.data)))
        res.append(str(self.data))
        return '\n'.join(res)

    @property
    def mz(self):
        """Get m/z values as a numpy array"""
        res = self._df.dropna().index.values
        return res

    @property
    def all_mz(self):
        """Get m/z values as a numpy array, even for missing data"""
        res = self._df.index.values
        return res

    def fillna(self, value):
        """Get new Spectrum with missing values replaced for a given value."""
        new_df = self._df.fillna(value)
        return Spectrum(df=new_df,
                        sample_name=self.sample_name,
                        label=self.label)

    def to_csv(self, filename, mz_name=None, sep=None, **kwargs):
        header = True
        if mz_name is not None:
            header = [mz_name] + list(self._df.columns)
        if sep is None:
            sep = '\t'
        self._df.to_csv(filename, header=header, sep=sep, index=True, **kwargs)
    
    def mz_to_csv(self, filename, mz_name=None, **kwargs):
        outdf = self.data.copy()
        outdf.index.name = mz_name
        outdf.to_csv(filename, columns=[])


class AlignedSpectra(object):
    """A set of peak lists, sharing m/z values.

    The underlying data (property `data`) is implemented in a Pandas DataFrame.
    The first column holds m/z values and the other columns hold (ususally)
    intensity data.

    Attributes
    ----------
    sample_names : list of str
        The names of the samples.
    labes : list of str
        Optional sample labels, used in classification tasks.

    """

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
                    self.sample_names = list(self._df.columns[:s_loc])
                else:
                    self.sample_names = list(self._df.columns)


    @property
    def data(self):
        """The Pandas DataFrame holding the MS data."""
        return self._df

    @property
    def sample_count(self):
        """Get the number of samples."""
        return len(self.sample_names)

    @property
    def mz(self):
        """Get m/z values as a numpy array"""
        res = self.data.index.values
        return res

    def __str__(self):
        return '\n'.join(
               ('{} samples:'.format(self.sample_count),
               str(self.sample_names),
               'Labels: {}'.format(self.labels),
               'Size: {}'.format(len(self.data)),
               str(self.data))
               ) # No, I will not become a JS freak


    def label_of(self,sample):
        """Get label from sample name"""
        if self.labels is None:
            return None
        for s, lbl in zip(self.sample_names, self.labels):
            if s == sample:
                return lbl
        return None

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
            out_df.to_csv(filename, header=True, sep=sep, index=True, **kwargs)
        else:
            # prepend output with result of header_func
            needs_to_close = False
            if _is_string(filename):
                of = open(filename, 'w') 
                needs_to_close = True
            else:
                of = filename
            
            header = header_func(self) + '\n'
            of.write(header)
            out_df.to_csv(of, header=False, sep=sep, 
                          index=True, **kwargs)
            
            if needs_to_close:
                of.close()
        
    def fillna(self, value):
        """Substitute missing values for a given value."""
        return AlignedSpectra(df=self._df.fillna(value),
                              sample_names=self.sample_names,
                              labels=self.labels)

    def sample(self, sample):
        """Get data for a given sample name, as a Spectrum."""
        df = self.data[[sample]].dropna()
        return Spectrum(df=df, sample_name=sample, label=self.label_of(sample))

    def unfold(self):
        """Return a list of Spectrum objects, unfolding the peak lists."""
        return [self.sample(name) for name in self.sample_names]

    def label(self, label):
        """Get data for a given label, as an AlignedSpectra object."""
         # build a list grouping samples with a given label
        lcolumns = []
        for c, l in zip(self.data.columns, self.labels):
            if l == label:
                lcolumns.append(c)
        df = self.data[lcolumns].dropna(how='all', subset=lcolumns)
        return AlignedSpectra(df=df, sample_names=lcolumns,
                                     labels=[label]*len(lcolumns))

    def common_label_mz(self, label1, label2):
        mz1 = self.label(label1).mz
        mz2 = self.label(label2).mz
        u = np.intersect1d(mz1, mz2)
        return u
    
    def exclusive_label_mz(self, label):
        # build list of unique other labels
        slabels = []
        for lbl in self.labels:
            if lbl not in slabels:
                slabels.append(lbl)
        slabels = [lbl for lbl in slabels if lbl != label]

        remaining = self.label(label).mz
        for lbl in slabels:
            remaining = np.setdiff1d(remaining, self.label(lbl).mz)
        return remaining

def read_spectrum(filename, label=None):
    s = pd.read_table(filename, index_col=False)
    # keep only the first two columns
    s = s.iloc[:, [0,1]]
    s = s.set_index(s.columns[0])
    return Spectrum(s, label=label)


def read_aligned_spectra(filename, labels=None):
    s = pd.read_table(filename, index_col=False)
    s = s.set_index(s.columns[0])
    return AlignedSpectra(s, labels=labels)


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
        print ('------ Reading MS-Excel file -------\n{}'.format(file_name))

    for sheetname in wb.sheet_names():
        if verbose:
            print ('- reading sheet "{}"...'.format(sheetname))

        # if sample_names argument if present (not None) then
        # if an integer, read row with sample names,
        # otherwise sample_names must a list of names for samples
        snames = []
        if sample_names is not None:
            if isinstance(sample_names, six.integer_types):
                sh = wb.sheet_by_name(sheetname)
                snames = sh.row_values(sample_names - 1)
                snames = [s for s in snames if len(s.strip()) > 0]
                header = sample_names
            else:
                snames = sample_names

        # read data (and discard empty xl columns)
        df = pd.read_excel(file_name,
                           sheetname=sheetname,
                           header=header)
        df = df.dropna(axis=1, how='all')

##         df.info()
##         print('============================================')
##         print(df.head())

        # if sample names were not set yet then
        # use "2nd columns" headers as sample names
        # if common_mz then use headers from position 1
        if len(snames) > 0:
            sample_names = snames
        else:
            if common_mz:
                sample_names = df.columns[1:]
            else:
                sample_names = df.columns[1::2]

        # split in groups of two (each group is a spectrum)
        results = []
        if not common_mz:
            j = 0
            for i in range(0, len(df.columns), 2):
                spectrum = df.iloc[:, i: i+2]
                spectrum.index = range(len(spectrum))
                spectrum = spectrum.dropna()
                spectrum = spectrum.set_index(spectrum.columns[0])
                spectrum = Spectrum(spectrum, sample_name=sample_names[j])
                results.append(spectrum)

                j = j + 1
        else:
            j = 0
            for i in range(1, len(df.columns)):
                spectrum = df.iloc[:, [0, i]]
                spectrum.index = range(len(spectrum))
                spectrum = spectrum.dropna()
                spectrum = spectrum.set_index(spectrum.columns[0])
                spectrum = Spectrum(spectrum, sample_name=sample_names[j])
                results.append(spectrum)

                j = j + 1

        if labels is not None:
            for i, spectrum in enumerate(results):
                spectrum.label = labels[i]

        if verbose:
            for spectrum in results:
                name = spectrum.sample_name
                print ('{:5d} peaks in sample {}'.format(spectrum.data.shape[0], name))
        spectra_table[sheetname] = results

    return spectra_table



def _sample_data():
    return """m/z	s38	s39	s40	s32	s33	s34
97.58868	1073218	1049440	1058971	2351567	1909877	2197036
97.59001	637830	534900	582966	1440216	1124346	1421899
97.59185	460092	486631		1137139	926038	1176756
97.72992			506345			439583
98.34894	2232032	2165052	1966283			
98.35078	3255288	2813578	2516386			
98.35122		2499163	2164976			
98.36001			1270764	1463557	1390574	
98.57354				4627491	6142759	
98.57382		3721991	3338506			4208438
98.57497	6229543	3347404	2327096			
98.57528				2510001	1989197	4377331
98.57599	6897403	3946118				
98.57621			3242232	2467520		4314818
98.57692	8116811	5708658	3899578			
98.57712				2418202	986128	4946201
98.57790	3891025	3990442	3888258	2133404		3643682
98.57899		1877649	1864650	1573559		1829208
99.28772	2038979				3476845	"""


if __name__ == '__main__':
    from six import StringIO

    sample = StringIO(_sample_data())
   
    print('Reading one spectrum (from aligned reads only first) ------------')
    spectrum = read_spectrum(sample)
    print(spectrum,'\n')
    spectrum.data.info()

    print('\nm/z values (excluding missing data) ----------')
    print(spectrum.mz)

    print('\Spectrum with missing values filled with zeros ----------')
    spectrumzero = spectrum.fillna(0)
    print(spectrumzero.data)

    print('\nSaving  mz into a file ----------')
    mzfile = StringIO()    
    spectrum.mz_to_csv(mzfile, mz_name='moverz')
    print('\n--- Resulting file:')
    print(mzfile.getvalue())

    print('\nReading aligned spectra (all of them) ----------')
    labels=['v1', 'v1', 'v1', 'v2', 'v2', 'v2']
    sample.seek(0)
    spectra = read_aligned_spectra(sample, labels=labels)
    print(spectra,'\n')
    spectra.data.info()

    print('\nData of sample s38 ----------')
    print(spectra.sample('s38'))
    
    print('\nm/z of sample s38 ----------')
    print(spectra.sample('s38').mz)
    
    print('\nData of label v1 ----------')
    print(spectra.label('v1'))
    
    print('\nm/z of label v1 ----------')
    print(spectra.label('v1').mz)

    print('\Spectra with missing values filled with zeros ----------')
    spectrazero = spectra.fillna(0)
    print(spectrazero.data)

    print('\nUnfolding spectra ----------')
    unfolded = spectra.unfold()
    
    for u in unfolded:
        print(u)
        print('+++++')
    
    print('\nSaving aligned spectra into a file ----------')
    def header(s):
        # this returns a header suitable for various metabolomics tools
        line1 = 'Labels,{}'.format(','.join(s.labels))
        line2 = 'Samples,'+','.join(s.sample_names)
        return '\n'.join([line1, line2])

    samplefile = StringIO()    
    spectra.to_csv(samplefile, header_func=header, sep=',')
    print('\n--- Resulting file:')
    print(samplefile.getvalue())
    #spectra.to_csv('testout.txt', header_func=header, sep=',')

    print('\nCommon m/z between labels (v1,v2) ----------')
    print(spectra.common_label_mz('v1', 'v2'))

    print('\nm/z values exclusive to label v1 ----------')
    print(spectra.exclusive_label_mz('v1'))

    print('\nComputing similarity measures ----------')
    
    sim = compute_similarity_measures(spectra)
    
    print('\n- Sample similarity --')
    print(sim.sample_similarity)
        
    print('\n- Label similarity --')
    print(sim.label_similarity)
