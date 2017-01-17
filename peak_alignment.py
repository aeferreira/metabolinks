from __future__ import print_function

import pandas as pd
from xlsxwriter.utility import xl_rowcol_to_cell, xl_col_to_name

def read_spectra_from_xcel(book_name, sheet_name, locs, skiprows=1):
    dfs = {}
    for k in locs:
        df = pd.read_excel(book_name, skiprows=skiprows, 
                           parse_cols=locs[k], 
                           sheetname=sheet_name)
        dfs[k] = df.dropna()
        dfs[k].columns = ['m/z', 'I'] # Force these column labels
    
    print ('\nProcessing')
    print ('- file "{}":'.format(book_name))
    print ('- sheet "{}":'.format(sheet_name))
    for k in dfs:
        print ('{:5d} peaks in sample {}'.format(dfs[k].shape[0], k))
    return dfs


def concat_peaks(base_dfs, verbose=True):
    if verbose:
        print ('- Joining data...', end=' ')
    dfs = {}
    # tag with sample names, concat vertically and sort by m/z
    for k in base_dfs:
        newdf = base_dfs[k].copy()
        newcol = pd.Series(k, index=newdf.index, name='_sample')
        newdf = pd.concat([newdf, newcol], axis=1)
        
        dfs[k] = newdf
    mdf = pd.concat([dfs[d] for d in dfs])
    mdf = mdf.sort_values(by='m/z')
    #reindex with increasing integers
    mdf.index = list(range(len(mdf)))
    
    if verbose:
        print('done, {} peaks before merging'.format(mdf.shape[0]))
    return mdf


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


def group_peaks(df, sample_ids, ppmtol=1.0, verbose=True):
    """Inserts a new column with integers indicating if entries belong to the same compound."""
    if verbose:
        print ('- Merging...', end=' ')
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
    
    result = pd.concat([df, 
                        pd.Series(glabels, index=df.index, name='_group')], 
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
            i = int(tagged.loc[s, 'I']) if s in tagged.index else 0
            intensities[s].append(i)
    
    result = pd.DataFrame(intensities, columns=colnames)
    if verbose:
        print('done, {} peaks after merging'.format(len(result)))
    return result


def save_to_excel(out_name, outdict, aligned_dir=True):
    print ('\n- writing to Excel workbook...', end=' ')    
    if aligned_dir:
        out_name = 'aligned/' + out_name
    writer = pd.ExcelWriter(out_name, engine='xlsxwriter')
    
    for k in outdict:
        results = outdict[k]
        n_compounds, ncols = results.shape
        #print 'CONTROL: n compounds =', n_compounds

        results.to_excel(writer, sheet_name=k, index=False, startrow=1, startcol=1)

        workbook  = writer.book
        worksheet = writer.sheets[k]

        # Center '# replicas' column
        format = workbook.add_format({'align': 'center'})
        worksheet.set_column(ncols, ncols, None, format)

        # Change displayed precision in 'm/z' column
        format2 = workbook.add_format({'num_format': '#.00000'})
        worksheet.set_column(1,1, None, format2)

        # Change widths
        worksheet.set_column(2, ncols-1, 25)
        worksheet.set_column(1,1, 15)
        worksheet.set_column(ncols,ncols, 15)

        # Create table of replica counts
        worksheet.write_string(1, ncols+2, '#samples')
        
        possible_counts = range(1, ncols-1)
        worksheet.write_column(2, ncols+2, possible_counts)
        
        counts_col = xl_col_to_name(ncols+2)
        replicas_col = xl_col_to_name(ncols)
        formula = '=COUNTIF(${}$3:${}${},{}{})'
        count_formulas = [formula.format(replicas_col, 
                                         replicas_col, 
                                         n_compounds+2, 
                                         counts_col, 
                                         i+2) for i in possible_counts]
        worksheet.write_column(2, ncols+3, count_formulas)

        worksheet.write_string(ncols, ncols+2, 'total')
        counts_col = xl_col_to_name(ncols+3)
        counts_range = '${}$3:${}${}'.format(counts_col, counts_col, ncols)
        worksheet.write_formula(ncols, ncols+3, '=SUM({})'.format(counts_range))

        # Add pie chart of replica counts
        chart = workbook.add_chart({'type': 'pie'})
        chart.add_series({'values': "='{}'!{}".format(k, counts_range), 'name':'#samples'})
        # chart.add_series({'values': '={}'.format(counts_range), 'name':'# replicas'})
        chart.set_size({'width': 380, 'height': 288})
        worksheet.insert_chart(7, ncols+1, chart)

    writer.save()
    print('done.\nCreated file "{}"'.format(out_name))


def workflow(excel_in_name, excel_out_name, sheetsnames, locs, 
             ppmtol, skiprows=1, aligned_dir=True):
    outdfs = {}
    for s in sheetsnames:
        dfs = read_spectra_from_xcel(excel_in_name, s, locs, skiprows=skiprows)
        mdf = concat_peaks(dfs)
        results = group_peaks(mdf, list(dfs.keys()), ppmtol=ppmtol)
        for n, c in results['#samples'].value_counts().iteritems():
            print('{} peaks in {} samples'.format(c,n))


        #print ('\nAligned data:')
        #print(results.head(10))

        outdfs[s] = results
    save_to_excel(excel_out_name, outdfs, aligned_dir)    

if __name__ == '__main__':
    ppmtol = 1.0
    
    indir = "vitis_fractions/"
    infiles = ['ESIneg_replicates_vitis_feb2016_recal.xlsx'] 
    #, 'ESIpos_replicates_vitis_feb2016.xlsx']
    
    snames = ['ACN', 'H2O', 'MeOH', 'Org']
    locs = {'23': 'B,C', '24': 'E,F', '25': 'H,I'}
    
    for fname in infiles:
        excel_in_name = indir + fname
        excel_out_name = indir + 'aligned_' + fname
        workflow(excel_in_name, excel_out_name, 
                 snames, locs, ppmtol, aligned_dir=False)
