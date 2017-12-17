from __future__ import print_function

from collections import OrderedDict
from numpy import nan

import six
import pandas as pd

from metabolinks.spectra import Spectrum, AlignedSpectra


def read_spectra_from_xcel(file_name,
                           sample_names=None,
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
                spectrum = Spectrum(spectrum, sample_name=sample_names[j])
                results.append(spectrum)

                j = j + 1
        else:
            j = 0
            for i in range(1, len(df.columns)):
                spectrum = df.iloc[:, [0, i]]
                spectrum.index = range(len(spectrum))
                spectrum = spectrum.dropna()
                spectrum = Spectrum(spectrum, sample_name=sample_names[j])
                results.append(spectrum)

                j = j + 1

        if verbose:
            for spectrum in results:
                name = spectrum.sample_name
                print ('{:5d} peaks in sample {}'.format(spectrum.data.shape[0], name))
        spectra_table[sheetname] = results

    return spectra_table


def concat_peaks(spectra, verbose=True):
    if verbose:
        print ('- Joining data...')
    dfs = []
    # tag with sample names, concat vertically and sort by m/z
    for spectrum in spectra:
        newdf = spectrum.data.copy()
        newdf.columns = ['m/z', 'I']
        newcol = pd.Series(spectrum.sample_name, index=newdf.index, name='_sample')
        newdf = pd.concat([newdf, newcol], axis=1)

        dfs.append(newdf)
    cdf = pd.concat(dfs)
    cdf = cdf.sort_values(by='m/z')
    #reindex with increasing integers
    cdf.index = list(range(len(cdf)))

    if verbose:
        print('  done, {} peaks in {} samples'.format(cdf.shape[0], len(spectra)))
    return cdf


def are_near(d1, d2, ppmtol):
    """Predicate: flags if two entries should belong to the same compound"""
    reltol = ppmtol * 1.0e-6
    # two consecutive peaks from the same sample
    # should not belong to the same group
    if d1['_sample'] == d2['_sample']:
        return False
    m1, m2 = d1['m/z'], d2['m/z']
    if (m2 - m1) / m2 <= reltol:
        return True
    return False


def group_peaks(df, sample_ids, ppmtol=1.0,
                min_samples=1, fillna=None,
                verbose=True):
    """Group peaks from different samples.

       Peaks are grouped if their relative mass difference is below
       a mass difference tolerance, in ppm."""

    if verbose:
        print ('- Aligning...')
    glabel = 0
    start = 0
    glabels = [0]

    for i in range(len(df)):
        if i == start:
            continue
        d1 = df.iloc[start]
        d2 = df.iloc[i]
        assert isinstance(d1['m/z'], (int, float))
        assert isinstance(d2['m/z'], (int, float))

        if not are_near(d1, d2, ppmtol):
            glabel += 1
            start = i

        glabels.append(glabel)

    result = pd.concat([df, pd.Series(glabels, index=df.index, name='_group')],
                        axis=1)
    grouped = result.groupby('_group')

    colnames = ['m/z'] + sample_ids + ['#samples', 'range_ppm']
    intensities = {sname: list() for sname in colnames}

    for gname, gvalue in grouped:
        tagged = gvalue.set_index('_sample')

        intensities['m/z'].append(tagged['m/z'].mean())
        intensities['#samples'].append(len(tagged))
        m_min = tagged['m/z'].min()
        m_max = tagged['m/z'].max()
        range_ppm = 1e6 * (m_max - m_min) / m_max
        intensities['range_ppm'].append(range_ppm)

        for s in sample_ids:
            i = int(tagged.loc[s, 'I']) if s in tagged.index else nan
            intensities[s].append(i)

    result = pd.DataFrame(intensities, columns=colnames)
    n_non_discarded = len(result)

    if verbose:
        print('  done, {} aligned peaks'.format(n_non_discarded))

    if min_samples > 1:
        result = result[result['#samples'] >= min_samples]

    if verbose:
        if min_samples > 1:
            n_discarded = n_non_discarded - len(result)
            print('- {} peaks were discarded (#samples < {})'.format(n_discarded, min_samples))

        counts_table = result['#samples'].value_counts().sort_index()
        for n, c in counts_table.iteritems():
            print ('{:5d} peaks in {} samples'.format(c, n))
        print('  {:3d} total'.format(len(result)))

    res = AlignedSpectra(result, sample_names=sample_ids)
    if fillna is not None:
        res.fillna(fillna)
    return res


def align_spectra(spectra,
                  ppmtol=1.0,
                  min_samples=1,
                  fillna=None,
                  verbose=True):

    samplenames = [s.sample_name for s in spectra]
    if verbose:
        print ('  Sample names:', samplenames)
    mdf = concat_peaks(spectra, verbose=verbose)
    return group_peaks(mdf, samplenames,
                       ppmtol=ppmtol,
                       min_samples=min_samples,
                       fillna=fillna,
                       verbose=verbose)



def save_aligned_to_excel(fname, outdict):
    writer = pd.ExcelWriter(fname, engine='xlsxwriter')

    for sname in outdict:
        results = outdict[sname].data
        n_compounds, ncols = results.shape

        # write Pandas DataFrame
        results.to_excel(writer, sheet_name=sname,
                         index=False, startrow=1, startcol=1)

        workbook = writer.book
        sh = writer.sheets[sname]

        # center '#samples' column
        format = workbook.add_format({'align': 'center'})
        sh.set_column(ncols, ncols, None, format)

        # change displayed precision in 'm/z' column
        format2 = workbook.add_format({'num_format': '#.00000'})
        sh.set_column(1, 1, None, format2)

        # change widths
        sh.set_column(2, ncols - 1, 25)
        sh.set_column(1, 1, 15)
        sh.set_column(ncols, ncols, 15)

        # Create table of replica counts
        sh.write_string(1, ncols + 2, '#samples')
        counts = results['#samples'].value_counts().sort_index()
        sh.write_column(2, ncols + 2, counts.index)
        sh.write_column(2, ncols + 3, counts.values)

        sh.write_string(2 + len(counts), ncols + 2, 'total')
        sh.write_number(2 + len(counts), ncols + 3, len(results))

        # Add pie chart of replica counts
        chart = workbook.add_chart({'type': 'pie'})
        chart.add_series({'values': [sname, 2, ncols + 3, 1 + len(counts), ncols + 3],
                          'categories': [sname, 2, ncols + 2, 1 + len(counts), ncols + 2],
                          'name': '#samples'})
        chart.set_size({'width': 380, 'height': 288})
        sh.insert_chart(7, ncols + 1, chart)

    writer.save()
    print('Created file\n{}'.format(fname))


def align_spectra_in_excel(fname, save_files_prefix=None,
                           save_to_excel=None,
                           ppmtol=1.0,
                           min_samples=1,
                           sample_names=None,
                           header_row=1,
                           fillna=None,
                           verbose=True):

    spectra_table = read_spectra_from_xcel(fname,
                                           sample_names=sample_names,
                                           header_row=header_row,
                                           verbose=verbose)

    if verbose:
        print('\n------ Aligning spectra ------')

    aligned_spectra = OrderedDict()
    for sheetname, spectra in spectra_table.items():
        if verbose:
            print ('\n-- sheet "{}"'.format(sheetname))
        aligned_spectra[sheetname] = align_spectra(spectra,
                                              ppmtol=ppmtol,
                                              min_samples=min_samples,
                                              fillna=fillna,
                                              verbose=verbose)
    if save_to_excel is not None:
        save_aligned_to_excel(save_to_excel, aligned_spectra)
    
    if save_files_prefix is not None:
        for sheetname, spectra in aligned_spectra.items():
            ofname = save_files_prefix + sheetname + '.txt'
            spectra.to_csv(ofname)
            if verbose:
                print('Created file\n{}'.format(ofname))

    return aligned_spectra


if __name__ == '__main__':
    ppmtol = 1.0
    min_samples = 1

    fname = '../example_data/data_to_align.xlsx'
    out_fname = '../example_data/aligned_data.xlsx'
    oprefix = '../example_data/aligned'

    header_row = 2
    sample_names = ['S1', 'S2', 'S3']
    #sample_names = 1

    align_spectra_in_excel(fname, save_files_prefix=oprefix,
                           save_to_excel=out_fname,
                           ppmtol=ppmtol,
                           min_samples=min_samples,
                           header_row=header_row,
                           fillna=0,
                           sample_names=sample_names)