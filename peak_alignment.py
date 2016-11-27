from __future__ import print_function
import pandas as pd
from xlsxwriter.utility import xl_rowcol_to_cell, xl_col_to_name

def read_spectra_from_xcel(book_name, sheet_name, locs, skiprows = 1):
    print ('\n- Reading data frames ---------------')
    dfs = {}
    for k in locs:
        df = pd.read_excel(book_name, skiprows = skiprows, parse_cols = locs[k], sheetname=sheet_name)
        dfs[k] = df.dropna()
        dfs[k].columns = ['m/z', 'I'] # Force these column labels
    print ('File "{}":'.format(book_name))
    print ('Sheet"{}":'.format(sheet_name))
    for k in dfs:
        print (k, '| shape =', dfs[k].shape)
    return dfs


def tag_dfs(base_dfs):
    print ('\n- Tagging with replicate names ------')
    dfs = {}
    for k in base_dfs:
        newdf = base_dfs[k].copy()
        newcol = pd.Series(k, index=newdf.index, name='tag')
        newdf = pd.concat([newdf, newcol], axis=1)
        
        dfs[k] = newdf
    print ('Data were tagged (with new col):')
    for k in dfs:
        print (k, '| shape =', dfs[k].shape)
    return dfs


def merge_spectra(dfs):
    print ('\n- Merging ---------------------------')
    mdf = pd.concat([dfs[d] for d in dfs])
    mdf = mdf.sort_values(by='m/z')
    #reindex with increasing integers
    mdf.index = range(len(mdf))
    print ('Data frames were merged. | shape = {}'.format(mdf.shape))
    return mdf


def are_near(d1,d2, ppmtol):
    """Predicate: flags if two entries should belong to the same compound"""
    reltol = ppmtol * 1.0e-6
    # two consecutive peaks from the same replica and fraction cannot belong to the same compound
    if d1['tag'] == d2['tag']:
        return False
    m1, m2 = d1['m/z'], d2['m/z']
    if (m2-m1) / m2 <= reltol:
        return True
    return False


def insert_groups_column(df, ppmtol=1.0):
    """Inserts a new column with integers indicating if entries belong to the same compound."""
    print ('- Inserting group labels ------------')
    glabel = 0
    start = 0
    glabels = [glabel]
    for i in range(len(df)):
        if i == start:
            continue
        d1 = df.iloc[start]
        d2 = df.iloc[i]
        if not isinstance(d1['m/z'], (int,float)) or not isinstance(d2['m/z'], (int,float)):
            print ('###### ERROR ############')
            #print 'in df', df.replica_name
            print (d1['m/z'], type(d1['m/z']))
            print (d2['m/z'], type(d2['m/z']))
            print ('start = ', start)
            print ('i = ', i)

        if not are_near(d1,d2, ppmtol):
            glabel += 1
            start = i
            
        glabels.append(glabel)
    glabels = pd.Series(glabels, index=df.index, name='group')
    result = pd.concat([df, glabels], axis=1)

    print ('Classification column inserted!')
    print ('Dataframe shape = {}, last group: {}'.format(result.shape, glabel))
    return result


def group_compounds(mdf):
    print ('\n- Grouping --------------------------')
    grouped = mdf.groupby('group')
    print ('Alignment done! # groups (compounds) = {}'.format(len(grouped)))
    return grouped


def moz_averages(g):
    return g.mean()['m/z']


def count_replicates(g):
    return g.size()


def split_I(g, ks):
    print ('\n- Splitting intensities -------------')
    Is = {}
    for k in ks:
        Is[k] = list()

    for gname, gvalue in g:
        tags = gvalue.set_index('tag')
        for k in ks:
            if k in tags.index and not isinstance(tags.loc[k, 'I'], (int,float)):
                print ('###### ERROR ############')
                print ('GROUP:', gname)
                print (tags)
                print ('----------------- k = ', k)
                print ('Intensity:')
                print (tags.loc[k, 'I'])
            i = int(tags.loc[k, 'I']) if k in tags.index else 0
            Is[k].append(i)
    for k in ks:
        Is[k] = pd.Series(Is[k], index=g.size().index)
    print ('Done!')
    return Is


def assemble_results(mean_mz, Is, n_replicas, filter_singles=False):
    print ('\n- Assembling results ----------------')
    s_list = [mean_mz]
    ks = sorted(Is.keys())
    for k in ks:
        s_list.append(Is[k])
    s_list.append(n_replicas)

    results = pd.concat(s_list, axis=1)
    results.columns = ['m/z'] + ["I - %s"%k for k in ks] + ['# replicas']
    if filter_singles:
        results = results[results['# replicas']>1]
    
    #reindex with increasing integers
    results.index = range(len(results))
    print ('Results data frame created.')
    
    return results


def save_to_excel(out_name, outdict, aligned_dir=True):
    print ('\n- Writing to Excel workbook ---------')    
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
        worksheet.write_string(1, ncols+2, '# replicas')
        
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
        chart.add_series({'values': "='{}'!{}".format(k, counts_range), 'name':'# replicas'})
        # chart.add_series({'values': '={}'.format(counts_range), 'name':'# replicas'})
        chart.set_size({'width': 380, 'height': 288})
        worksheet.insert_chart(7, ncols+1, chart)

    writer.save()
    print ('------->>>>>>>>>>> Results saved to file {}.'.format(out_name))


def workflow(excel_in_name, excel_out_name, sheetsnames, locs, 
             ppmtol, skiprows=1, aligned_dir=True):
    outdfs = {}
    for s in sheetsnames:
        print ('=========================================================')
        print ('Processing sheet {}, form file {}\n'.format(s, excel_in_name))
        
        dfs = read_spectra_from_xcel(excel_in_name, s, locs, skiprows=skiprows)
        dfs_tagged = tag_dfs(dfs)
        mdf = merge_spectra(dfs_tagged)
        mdf_g = insert_groups_column(mdf, ppmtol=ppmtol)
        grouped = group_compounds(mdf_g)
        n_compounds = len(grouped)
        mean_mz = moz_averages(grouped)
        n_replicas = count_replicates(grouped)
        ks = sorted(dfs.keys())
        Is = split_I(grouped, ks)
        results = assemble_results(mean_mz, Is, n_replicas)

        print ('=========================================================')
        print ('\nProcessed sheet {}, form file {}'.format(s, excel_in_name)) 
        # print '\nFinal data frame shape:{}'.format(results.shape)
        print ('\nFinal data frame:')
        results.info()

        outdfs[s] = results
    save_to_excel(excel_out_name, outdfs, aligned_dir)    
