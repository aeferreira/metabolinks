"""Classes representing MS data."""

import six

import numpy as np
import pandas as pd

from metabolinks import dataio
from metabolinks.utils import _is_string
from metabolinks.similarity import mz_similarity


class MSDataSet(object):
    """A general container for MS data.

    This is a wrapper around a Pandas DataFrame (the `data` property).

    Direct metadata is stored in the (hierarquical) column index of `data`:
    The (row) index contains one level, the "features", often labels of spectral entities. Examples are
    m/z values, formulae or any format-specific labeling scheme.
    
    Columns contain three levels (outwards from the data):

    - `info_type` describes the type of data in each column. The default is "I". A common example is
       the pair "RT", "I" for each sample
    - `sample` contains the names of samples
    - `label` contains the labels of each sample
    
    In the display of data, if the all entries of a given level are equal then that level will
    be absent from the output.

    """

    def __init__(self, data, features=None, samples=None, labels=None, info_types=None, features_name=None, info_name=None):
        """Build member `_df`, aliased as `data` as a pandas DataFrame with appropriate index and columns from data and keyword arguments.

        Information is retrieved from `data` function argument, keyword arguments may overwrite it and, if needed,
        default values are provided.
        `data` function argument can be a numpy array, a pandas DataFrames or structures with a DataFrame member `data`.
        """

        self._df = dataio.gen_df(data, features=features,
                                samples=samples,
                                labels=labels,
                                info_types=info_types,
                                features_name=features_name,
                                info_name=info_name)

    @property
    def data(self):
        """The Pandas DataFrame holding the MS data."""
        return self._df

    def __len__(self):
        return len(self.data)

    @property
    def labels(self):
        """Get the different data labels (no repetitions)."""
        il_labels = self.data.columns.names.index('label')
        return self.data.columns.levels[il_labels]

    @property
    def samples(self):
        """Get the different sample names."""
        il_sample = self.data.columns.names.index('sample')
        return self.data.columns.levels[il_sample]

    @property
    def features(self):
        """Get the features array."""
        return self.data.index.values.copy()

    @property
    def feature_count(self):
        """Get the number of features."""
        return len(self.data.index)

    @property
    def sample_count(self):
        """Get the number of samples."""
        return len(self.samples)

    @property
    def label_count(self):
        """Get the number of labels."""
        # 'no label' still counts as one (global) label
        return len(self.labels)

    @property
    def no_labels(self):
        """True if there is only one (global) label 'no label'."""
        return len(self.labels) == 1 and self.labels[0] == 'no label'

    # def set_labels(self, lbs=None):
    #     if lbs is None:
    #         self.labels = None
    #     elif _is_string(lbs):
    #         self.labels = [lbs] * len(self.sample_names)
    #     else:
    #         self.labels = list(lbs[:len(self.sample_names)])

    
    def __str__(self):
        return '\n'.join(
               ('{} samples:'.format(self.sample_count),
               str(list(self.samples)),
               'Labels: {}'.format(list(self.labels)),
               '{} features'.format(self.feature_count),
               str(self.data))
               ) # No, I will not become a js freak

    def info(self):
        res = ['Number of peaks: {}'.format(len(self.data))]
        for name in self.samples:
            sample = self.sample(name)
            label = sample.label
            peaks = len(sample)
            res.append('{:5d} peaks in sample {}, with label {}'.format(peaks, name, label))
        return '\n'.join(res)

    def label_of(self,sample):
        """Get label from sample name"""
        if self.labels is None:
            return None
        for s, lbl in zip(self.sample_names, self.labels):
            if s == sample:
                return lbl
        return None

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

    # def fillna(self, value):
    #     """Substitute missing values for a given value."""
    #     return MSDataSet(self.data.fillna(value),
    #                           sample_names=self.sample_names,
    #                           labels=self.labels)

    # def sample(self, sample):
    #     """Get data for a given sample name, as a Spectrum."""
    #     df = self.data[[sample]].dropna()
    #     return Spectrum(df, sample_name=sample, label=self.label_of(sample))

    # def unfold(self):
    #     """Return a list of Spectrum objects, unfolding the peak lists."""
    #     return [self.sample(name) for name in self.sample_names]

    # def label(self, label):
    #     """Get data for a given label, as an AlignedSpectra object."""
    #     # build a list grouping samples with a given label
    #     lcolumns = []
    #     for c, l in zip(self.data.columns, self.labels):
    #         if l == label:
    #             lcolumns.append(c)
    #     df = self.data[lcolumns].dropna(how='all', subset=lcolumns)
    #     return MSDataSet(df, sample_names=lcolumns,
    #                               labels=[label]*len(lcolumns))
    
    # def unique_labels(self):
    #     if self.labels is None:
    #         return None
    #     ulabels = []
    #     for lbl in self.labels:
    #         if lbl not in ulabels:
    #             ulabels.append(lbl)
    #     return ulabels
    
    # def rep_at_least(self, minimum=1):
    #     df = self._df.copy()
    #     # build list of unique labels
    #     unique_labels = list(set(self.labels))
    #     for label in unique_labels:
    #         # build a list grouping samples with each given label
    #         lcolumns = []
    #         for c, l in zip(self.data.columns, self.labels):
    #             if l == label:
    #                 lcolumns.append(c)
    #         lessthanmin = df[lcolumns].count(axis=1) < minimum
    #         df.loc[lessthanmin, lcolumns] = np.nan
    #     df = df.dropna(how='all')
    #     return MSDataSet(df, sample_names=self.sample_names,
    #                               labels=self.labels)

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

    data = pd.read_csv(sample, sep='\t').set_index('m/z')
    # print('-----------------------')
    # print(f'data = \n{data}')
    # print('-----------------------')
    dataset = MSDataSet(data)
    # print('dataset.data =')
    # print(dataset.data)
    # print('-----------------------')
    print('MSDataSet from string data (as io stream) ------------')
    print(dataset)

    # print('\nSpectrum with missing values filled with zeros ----------')
    # spectrumzero = spectrum.fillna(0)
    # print(spectrumzero.data)

    # print('\nSaving  mz into a file ----------')
    # mzfile = StringIO()    
    # spectrum.mz_to_csv(mzfile, mz_name='moverz')
    # print('\n--- Resulting file:')
    # print(mzfile.getvalue())

    # print('\nReading aligned spectra (all of them) no labels ----------')
    # sample.seek(0)
    # spectra = read_aligned_spectra(sample)
    # print(spectra,'\n')
    # spectra.data.info()
    # print(spectra.info())

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

    # print('\nData of sample s38 ----------')
    # print(spectra.sample('s38'))
    
    # print('\nm/z of sample s38 ----------')
    # print(spectra.sample('s38').mz)
    
    # print('\nData of label v1 ----------')
    # print(spectra.label('v1'))
    
    # print('\nm/z of label v1 ----------')
    # print(spectra.label('v1').mz)

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
    
##     print('\nReading from Excel ----------')
##     file_name='data_to_align.xlsx'
##     spectra3 = read_spectra_from_xcel(file_name,
##                            sample_names=1, header_row=1)
    
