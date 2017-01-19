from __future__ import print_function

from collections import OrderedDict
from numpy import nan

import pandas as pd
from xlsxwriter.utility import xl_rowcol_to_cell, xl_col_to_name

def read_spectra_from_xcel(file_name,
                           sample_names=None,
                           samples_row=None, 
                           header_row=1,
                           verbose=True):
    
    spectra_table = OrderedDict()

    wb = pd.ExcelFile(file_name).book
    header = header_row - 1
    
    if verbose:
        print ('------ Reading MS-Excel file -------\n{}'.format(file_name))

    for sheetname in wb.sheet_names():
        if verbose:
            print ('- reading sheet "{}"...'.format(sheetname))

        # use sample_names from function argument if present
        # otherwise read row with sample names, if present
        snames = []
        if sample_names is not None:
            snames = sample_names
        elif samples_row is not None:
            sh = wb.sheet_by_name(sheetname)
            snames = sh.row_values(samples_row-1)
            snames = [s for s in snames if len(s.strip()) > 0]
            header = samples_row

        # read data (and discard empty Xcel columns
        df = pd.read_excel(file_name, 
                           sheetname=sheetname,
                           header=header)
        df = df.dropna(axis=1, how='all')

##         df.info()
##         print('============================================')
##         print(df.head())
        
        # if there is not a row with sample names, or the function argument,
        # use "2nd columns" headers as sample names
        if len(snames) > 0:
            sample_names = snames
        else:
            sample_names = df.columns[1::2]
        
##         print('============================================')
##         print('Sample names:', sample_names)
        
        # split in groups of two (each group is a spectrum)
        results = []
        j = 0
        for i in range(0, len(df.columns), 2):
            spectrum = df.iloc[:, i:i+2]
            spectrum.index = range(len(spectrum))
            spectrum = spectrum.dropna()
            spectrum.columns = ['m/z', 'I'] # force these column labels
            results.append((sample_names[j], spectrum))

##             print('============================================')
##             print(snames_row[j])
##             print('-------------------------------')
##             print(spectrum.head(10))
            j = j + 1

        if verbose:
            for name, spectrum in results:
                print ('{:5d} peaks in sample {}'.format(spectrum.shape[0], name))
        spectra_table[sheetname] = results

    return spectra_table


def concat_peaks(spectra, verbose=True):
    if verbose:
        print ('- Joining data...')
    dfs = []
    # tag with sample names, concat vertically and sort by m/z
    for samplename, spectrum in spectra:
        newdf = spectrum.copy()
        newcol = pd.Series(samplename, index=newdf.index, name='_sample')
        newdf = pd.concat([newdf, newcol], axis=1)
        
        dfs.append(newdf)
    cdf = pd.concat(dfs)
    cdf = cdf.sort_values(by='m/z')
    #reindex with increasing integers
    cdf.index = list(range(len(cdf)))
    
    if verbose:
        print('  done, {} peaks in {} samples'.format(cdf.shape[0], len(spectra)))
    return cdf


def are_near(d1,d2, ppmtol):
    """Predicate: flags if two entries should belong to the same compound"""
    reltol = ppmtol * 1.0e-6
    # two consecutive peaks from the same sample should not belong to the same group
    if d1['_sample'] == d2['_sample']:
        return False
    m1, m2 = d1['m/z'], d2['m/z']
    if (m2-m1) / m2 <= reltol:
        return True
    return False


def group_peaks(df, sample_ids, ppmtol=1.0, min_samples=1, verbose=True):
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
        assert isinstance(d1['m/z'], (int,float))
        assert isinstance(d2['m/z'], (int,float))

        if not are_near(d1,d2, ppmtol):
            glabel += 1
            start = i

        glabels.append(glabel)
    
    result = pd.concat([df, pd.Series(glabels, index=df.index, name='_group')], 
                        axis=1)
    grouped = result.groupby('_group')

##     print('++++++++++++++++++++++++++++++++++++++++++++++')
##     for (i,(gname, gvalue)) in enumerate(grouped):
##         print ('----->>>{}, {} elements'.format(gname, len(gvalue)))
##         print (gvalue.set_index('_sample'))
##         if i > 6:
##             break
##     print('++++++++++++++++++++++++++++++++++++++++++++++')

    colnames = ['m/z'] + sample_ids + ['#samples']
    intensities = {sname:list() for sname in colnames}
    
    for gname, gvalue in grouped:
        tagged = gvalue.set_index('_sample')
        
        intensities['m/z'].append(tagged['m/z'].mean())
        intensities['#samples'].append(len(tagged))
        
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

    return result

def save_aligned_to_excel(fname, outdict):
    print ('\n------ writing results to MS-Excel file ------')    
    writer = pd.ExcelWriter(fname, engine='xlsxwriter')
    
    for sname in outdict:
        results = outdict[sname]
        n_compounds, ncols = results.shape

        # write Pandas DataFrame
        results.to_excel(writer, sheet_name=sname, 
                         index=False, startrow=1, startcol=1)

        workbook  = writer.book
        sh = writer.sheets[sname]

        # center '#samples' column
        format = workbook.add_format({'align': 'center'})
        sh.set_column(ncols, ncols, None, format)

        # change displayed precision in 'm/z' column
        format2 = workbook.add_format({'num_format': '#.00000'})
        sh.set_column(1, 1, None, format2)

        # change widths
        sh.set_column(2, ncols-1, 25)
        sh.set_column(1, 1, 15)
        sh.set_column(ncols, ncols, 15)

        # Create table of replica counts
        sh.write_string(1, ncols+2, '#samples')
        counts = results['#samples'].value_counts().sort_index()
        sh.write_column(2, ncols+2, counts.index)
        sh.write_column(2, ncols+3, counts.values)

        sh.write_string(2+len(counts), ncols+2, 'total')
        sh.write_number(2+len(counts), ncols+3, len(results))
        
        # Add pie chart of replica counts
        chart = workbook.add_chart({'type': 'pie'})
        chart.add_series({'values': [sname, 2, ncols+3, 1+len(counts), ncols+3],
                          'categories': [sname, 2, ncols+2, 1+len(counts), ncols+2],
                          'name':'#samples'})
        chart.set_size({'width': 380, 'height': 288})
        sh.insert_chart(7, ncols+1, chart)

    writer.save()
    print('Created file\n{}'.format(fname))


def align_spectra(spectra,
                  ppmtol=1.0, 
                  min_samples=1,
                  verbose=True):
    
    samplenames = [name for name, df in spectra]
    mdf = concat_peaks(spectra, verbose=verbose)
    return group_peaks(mdf, samplenames,
                       ppmtol=ppmtol,
                       min_samples=min_samples, 
                       verbose=verbose)

def align_spectra_in_excel(fname, save_to_excel=None, 
                           ppmtol=1.0, 
                           min_samples=1,
                           sample_names=None,
                           samples_row=None, 
                           header_row=1,
                           verbose=True):
    
    spectra_table = read_spectra_from_xcel(fname, 
                                           sample_names=sample_names,
                                           samples_row=samples_row,
                                           header_row=header_row,
                                           verbose=verbose)
    if verbose:
        print ('\n------ Aligning spectra ------')

    outspectra = OrderedDict()
    for sheetname, spectra in spectra_table.items():
        if verbose:
            print ('\n-- sheet "{}"'.format(sheetname))
        outspectra[sheetname] = align_spectra(spectra,
                                              ppmtol=ppmtol,
                                              min_samples=min_samples,
                                              verbose=verbose)
    if save_to_excel is not None:
        save_aligned_to_excel(save_to_excel, outspectra)    

    return outspectra
    

if __name__ == '__main__':
    ppmtol = 1.0
    min_samples = 1
        
    fname = 'vitis_fractions/ESIneg_replicates_vitis_feb2016_recal.xlsx'
    out_fname = 'vitis_fractions/aligned_ESIneg_replicates_vitis_feb2016_recal.xlsx'
    
    header_row = 2
    samples_row = 1
    sample_names = ['23', '24', '25']
    sample_names = None
    
    align_spectra_in_excel(fname, save_to_excel=out_fname,
                           ppmtol=ppmtol,
                           min_samples=min_samples,
                           header_row=header_row,
                           samples_row=samples_row,
                           sample_names=sample_names)