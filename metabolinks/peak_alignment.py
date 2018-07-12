from __future__ import print_function

import itertools
from collections import OrderedDict

import numpy as np
import pandas as pd

from metabolinks.spectra import AlignedSpectra, read_spectra_from_xcel, read_spectrum

def are_near(d1, d2, reltol):
    """Predicate: flags if two entries should belong to the same compound"""
    # two consecutive peaks from the same sample
    # should not belong to the same group
    if d1['_spectrum'] == d2['_spectrum']:
        return False
    m1, m2 = d1['m/z'], d2['m/z']
    if (m2 - m1) / m2 <= reltol:
        return True
    return False


def group_peaks(df, ppmtol=1.0):
    """Compute groups of peaks, according to m/z proximity."""
    
    # compute group indexes
    glabel = 0
    start = 0
    glabels = [0]
    
    reltol = ppmtol * 1.0e-6

    for i in range(len(df)):
        if i == start:
            continue
        d1 = df.iloc[start]
        d2 = df.iloc[i]
        assert isinstance(d1['m/z'], (int, float))
        assert isinstance(d2['m/z'], (int, float))

        if not are_near(d1, d2, reltol):
            glabel += 1
            start = i

        glabels.append(glabel)

    # insert new column with group indexes and group by this column
    result = pd.concat([df, pd.Series(glabels, index=df.index, name='_group')],
                        axis=1)

    return result.groupby('_group')
    
def align(inputs, ppmtol=1.0, min_samples=1, verbose=True):
    """Align peak lists, according to m/z proximity.
    
       Returns an instance of AlignedSpectra."""

    # Compute sample names and labels of the resulting table
    samplenames = []
    for s in inputs:
        try:
            samplenames.append([s.sample_name])
        except AttributeError as error:
            samplenames.append(s.sample_names)
    # flatten list to get all sample names
    
    all_samplenames = list(itertools.chain(*samplenames))

    no_labels = False
    for s in inputs:
        try:
            if s.label is None:
                no_labels = True
                break
        except AttributeError as error:
            if s.labels is None:
                no_labels = True
                break
    if no_labels:
        labels = None
    else:
        labels = []
        for s in inputs:
            try:
                labels.append(s.label)
            except AttributeError as error:
                labels.extend(s.labels)
    # Compute slices to use in table building
    ncols = 0
    tslices = []
    for namelist in samplenames:
        n = len(namelist)
        tslices.append((ncols, ncols+n))
        ncols += n
       
    if verbose:
        print ('------ Aligning spectra -------------')
        print ('  Sample names:', samplenames)
        if not no_labels:
            print ('  Labels:', labels)
            
        print ('- Joining data...', end=' ')
    
    # Joining data (vertically)
    
    n = len(inputs)
    dfs = []
    
    # tag with sample names, concat vertically and sort by m/z
    for sindx, s in enumerate(inputs):
        # create DataFrame with columns 'm/z', '_peak_index', and '_spectrum'
        dfs.append(
            pd.DataFrame({
                'm/z': s.mz,
                '_peak_index': range(len(s.data)),
                '_spectrum': sindx
            })
        )

    # sort by m/z
    cdf = pd.concat(dfs) # vertically
    cdf = cdf.sort_values(by='m/z')
    #reindex with increasing integers
    cdf.index = list(range(len(cdf)))

    if verbose:
        print('done, (total {} peaks in {} spectra)'.format(cdf.shape[0], n))
        
    # Grouping data and building resulting table (as an AlignedSpectra instance)

    if verbose:
        print ('- Aligning...', end=' ')

    grouped = group_peaks(cdf, ppmtol=ppmtol)
    
    colnames = ['m/z'] + all_samplenames + ['range_ppm']
    
    nan = np.nan
    aligned_array = np.full((len(grouped), len(all_samplenames)), nan)
    
    mz_array = [0.0] * len(grouped)
    mz_range_array = [0.0] * len(grouped)
    
    for igroup, (_, g) in enumerate(grouped):
        group = g.set_index('_spectrum')

        mz_array[igroup] = group['m/z'].mean()

        m_min, m_max = group['m/z'].min(), group['m/z'].max()
        range_ppm = 1e6 * (m_max - m_min) / m_min
        mz_range_array[igroup] = range_ppm

        for i, s in enumerate(inputs):
            if i in group.index:
                row = group.loc[i, '_peak_index']
                start, end = tslices[i][0], tslices[i][1]
                ndata = end-start
                aligned_array[igroup, start:end] = s.data.iloc[row, :ndata]

    result = pd.DataFrame(aligned_array, 
                          columns=all_samplenames, 
                          index=mz_array)
    # insert column with sample count
    result.insert(len(result.columns), 
                  '#samples',
                  result.count(axis=1))
    # insert column with ppm range
    result.insert(len(result.columns), 
                  '#range_ppm',
                  pd.Series(mz_range_array, index=result.index))

    # Discard rows according to min_samples cutoff
    n_non_discarded = len(result)

    if min_samples > 1:
        result = result[result['#samples'] >= min_samples]

    if verbose:
        print('done, {} aligned peaks'.format(n_non_discarded))

        if min_samples > 1:
            n_discarded = n_non_discarded - len(result)
            msg = '- {} peaks were discarded (#samples < {})'.format
            print(msg(n_discarded, min_samples))

        counts_table = result['#samples'].value_counts().sort_index()
        for n, c in counts_table.iteritems():
            print ('{:5d} peaks in {} samples'.format(c, n))
        #print('  {:3d} total'.format(len(result)))

    result = AlignedSpectra(result, 
                            sample_names=all_samplenames,
                            labels=labels)
    
    if verbose:
        print(result.info())

    return result

align_spectra = align

def align_spectra_in_excel(fname,
                           save_to_excel=None,
                           ppmtol=1.0, min_samples=1,
                           sample_names=None, labels=None,
                           header_row=1,
                           fillna=None,
                           verbose=True):

    spectra_table = read_spectra_from_xcel(fname,
                                           sample_names=sample_names,
                                           labels=labels,
                                           header_row=header_row,
                                           verbose=verbose)

    if verbose:
        print('\n------ Aligning spectra ------')

    aligned_spectra = OrderedDict()
    for sheetname, spectra in spectra_table.items():
        if verbose:
            print ('\n-- sheet "{}"'.format(sheetname))
        aligned = align(spectra,
                        ppmtol=ppmtol, min_samples=min_samples,
                        verbose=verbose)
        if fillna is not None:
            aligned = aligned.fillna(fillna)
        aligned_spectra[sheetname] = aligned

    if save_to_excel is not None:
        save_aligned_to_excel(save_to_excel, aligned_spectra)
    
    return aligned_spectra


def save_aligned_to_excel(fname, aligned_dict):
    writer = pd.ExcelWriter(fname, engine='xlsxwriter')

    for sname in aligned_dict:
        
        aligned_spectra = aligned_dict[sname]
        
        sim = aligned_spectra.compute_similarity_measures()
                
        sample_similarity = sim.sample_similarity
        label_similarity = sim.label_similarity
        unique_labels = sim.unique_labels
                
        results = aligned_spectra.data
        results = results.reset_index(level=0)
        
        n_compounds, ncols = results.shape

        # write Pandas DataFrame
        results.to_excel(writer, sheet_name=sname,
                         index=False, startrow=1, startcol=1)

        workbook = writer.book
        sh = writer.sheets[sname]

        # center '#samples' column
        format = workbook.add_format({'align': 'center'})
        sh.set_column(ncols-1, ncols-1, None, format)

        # change displayed precision in 'm/z' column
        format2 = workbook.add_format({'num_format': '0.0000000'})
        sh.set_column(1, 1, 15, format2)

        # change widths
        sh.set_column(2, ncols - 2, 25)
        #sh.set_column(1, 1, 15)
        format3 = workbook.add_format({'num_format': '0.0000'})
        sh.set_column(ncols, ncols, 15, format3)
        
        # Create report sheet
        rsheetname = sname + ' report'
        sh = workbook.add_worksheet(rsheetname)
        
        row_offset = 1
        
        sh.write_string(row_offset, 1, 'Peak reproducibility')
        
        row_offset = row_offset + 2
        
        # create table of replica counts
        sh.write_string(row_offset, 2, '#samples')
        counts = results['#samples'].value_counts().sort_index()
        sh.write_column(row_offset + 1, 2, counts.index)
        sh.write_column(row_offset + 1, 3, counts.values)

        sh.write_string(row_offset + 1 + len(counts), 2, 'total')
        sh.write_number(row_offset + 1 + len(counts), 3, len(results))

        # Add pie chart of replica counts
        chart = workbook.add_chart({'type': 'pie'})
        chart.add_series({'values': [rsheetname, row_offset+1, 3, row_offset + len(counts), 3],
                          'categories': [rsheetname, row_offset+1, 2, row_offset + len(counts), 2],
                          'name': '#samples'})
        chart.set_size({'width': 380, 'height': 288})
        sh.insert_chart(1, 5, chart)

        row_offset = row_offset + len(counts) + 7

        sh.write_string(row_offset, 1, 'Sample sizes')
        
        row_offset = row_offset + 2
        
        sh.write_row(row_offset, 1, aligned_spectra.sample_names)
        for i in range(sample_similarity.shape[0]):
            sh.write_number(row_offset + 1, 1 + i, sample_similarity[i, i])

        row_offset = row_offset + 4
        
        sh.write_string(row_offset, 1, 'Sample similarity')
        
        row_offset = row_offset + 2

        sh.write_row(row_offset, 2, aligned_spectra.sample_names)
        sh.write_column(row_offset + 1, 1, aligned_spectra.sample_names)
        for i in range(sample_similarity.shape[0]):
            sh.write_row(row_offset + i + 1, 2, sample_similarity[i, :])
            
        if label_similarity is not None:
            row_offset = row_offset + sample_similarity.shape[0] + 3
            
            sh.write_string(row_offset, 1, 'Label similarity')
            
            row_offset = row_offset + 2

            sh.write_row(row_offset, 2, unique_labels)
            sh.write_column(row_offset + 1, 1, unique_labels)
            for i in range(label_similarity.shape[0]):
                sh.write_row(row_offset + i + 1, 2, label_similarity[i, :])

    writer.save()
    print('Created file\n{}'.format(fname))


if __name__ == '__main__':
    
    def _sample1():
        return """m/z	S1
184.75927	401638
219.98397	372923
241.21758	516287
242.17641	2216562
243.96836	875562
244.98563	1851690
248.08001	436328
253.21779	590613
255.23318	4244258
256.23617	818943
265.14807	11979620
266.15147	1454566
267.14409	524795
269.24915	399273
279.16415	652112
280.98317	7396249
280.98775	1544346
281.98709	461478
283.26439	8264975
284.26768	1734495
293.17939	5848741
293.18427	1209990
294.18297	1011921
294.98292	876686
297.15321	2832829
298.15686	588668
309.17431	4808438
310.17737	646849
311.16876	8185572
312.1723	1570308
313.1651	470517
321.1743	2165299
321.21145	702015
322.1776	454479
325.18445	7327948
326.18754	1525032
335.1899	5498360
336.19345	1113508
337.20575	2005736
339.20004	4417087
340.20324	933858
341.27064	505890
344.97918	19448392
345.98282	1335231
351.18646	465893
353.20026	2748696
355.28529	609784
365.24646	1753079
366.30194	1118254
367.24251	563866
375.27556	1542938
380.97678	6282830
381.2311	1095822
381.9812	477255
393.2777	5599757
394.28125	1171962
394.97517	1076064
395.27512	1810532
396.27872	569001
397.22595	1030611
403.30669	4475977
403.31602	1056685
404.31025	1189038
420.25071	579067
420.29726	771210
422.97064	591497
429.29603	919938
437.24091	1106971
441.25223	525592
444.9727	22531764
445.97546	1949441
477.06725	1589463
554.26202	39856296
554.70336	658268
554.71575	670736
554.75188	608460
554.77597	694228
554.81207	671552
554.84877	678828
555.01794	633971
555.11506	743299
555.26474	12299251
556.26908	2195766
590.23599	673170
617.25588	1468270
618.26331	604977
"""

    def _sample2():
        return """m/z	S2
99.65818	415592
107.40354	409978
112.98545	385199
227.20227	430169
228.16055	552691
239.09298	463301
241.21753	665785
242.17642	6233384
243.17988	851608
243.96857	410324
253.21748	624711
255.23317	7923468
255.23693	1705901
256.23625	1479363
265.12471	730251
265.1481	15849385
266.104	414807
266.15158	2024276
267.1441	611025
269.24913	798384
270.20755	470073
276.62781	805495
279.16418	599787
281.24883	1008184
283.2644	17928410
284.26767	3473844
293.17939	6586083
294.18284	1032199
297.15326	3326474
298.15666	589246
309.17428	6911028
310.17761	896752
311.16874	10302186
312.17217	2002392
313.16523	460152
321.21121	759001
325.18445	9555216
326.18752	1764149
329.19212	522418
329.26887	500068
337.20564	2161277
339.19998	5779029
340.20317	1178719
341.27074	462657
353.2004	3750696
354.20302	531812
355.28523	793821
365.24641	2585171
366.25088	437329
367.24265	825054
372.30457	620391
374.24451	794178
375.27545	3172017
375.28335	745684
376.2795	655983
381.23119	1122553
390.29864	1206014
391.30131	499361
393.27766	8311432
394.28131	1748937
395.27498	2766184
395.28385	815900
396.27851	771052
397.2259	1430550
403.30653	10426832
404.31005	2866461
405.26597	487824
410.22039	759338
414.32366	590241
420.25059	1274043
420.2976	851531
425.25847	533120
437.2408	1185705
441.25239	615064
447.30839	591070
477.06722	3735876
478.07209	676633
536.24964	695616
554.26202	50734088
554.78846	739116
554.86066	728516
554.8972	682720
554.92137	694169
555.00545	712821
555.01813	729225
555.04226	667217
555.06616	817641
555.07834	759085
555.10236	774981
555.16303	1144889
555.26481	15089114
556.26894	2822178
557.26831	592422
566.26071	878810
571.16641	603572
590.23552	811060
617.25588	1050059
"""

    def _sample3():
        return """m/z	S3
99.65817	524063
107.40355	705610
110.86046	325953
112.72493	388959
209.086	372373
227.20199	851247
228.16057	509236
238.03059	441017
241.21749	1181213
242.17642	7013381
243.17987	928451
243.96857	435358
253.21761	2200491
254.2207	460452
255.23317	12199474
256.23639	2205541
265.14808	31571388
265.19941	591967
266.15146	4121529
267.14378	1254985
267.23322	599772
269.24897	1482466
270.20759	479167
277.13083	495438
279.16395	1139656
279.23275	728373
281.24878	3089253
282.2522	641011
283.26439	22737690
284.26763	4586268
293.17939	12157415
294.18293	2032992
295.17556	554809
297.15324	4791022
297.15825	1107151
298.15685	993103
301.23871	493007
309.17427	15774029
310.17766	2303012
311.1688	15580988
312.17228	2803691
313.16485	738146
321.21103	1211624
325.18441	15118850
326.18765	3093784
327.18054	599838
329.26904	569972
337.20558	5128711
338.20876	694241
339.20004	9590034
339.32726	510068
340.20338	1987440
341.19555	531758
341.2702	814098
353.20044	9608569
354.20369	1479680
355.19676	518384
355.28534	1146181
365.24653	4497228
366.25044	867335
367.24323	1461440
367.35791	765327
372.304	752886
374.24425	721405
375.27534	4692616
376.27922	968137
381.23162	2634808
382.23419	616498
389.29087	553121
390.29881	575616
393.27772	14317644
394.2812	3280305
395.27491	4659218
395.28385	1212758
396.27859	1273511
397.22642	3718687
398.2292	626614
403.30652	15574090
404.29342	1008207
404.31019	4017488
404.31935	926838
405.26608	517360
405.31174	525157
410.22064	1047792
414.32313	954825
420.25058	1500214
420.2971	1222623
425.25828	1124338
437.24071	1492111
441.25279	1324302
447.30849	622852
447.3344	572952
477.06726	4784492
477.08089	1062740
478.07182	1145917
490.22824	677982
536.24885	606481
553.74744	614514
553.75968	620166
554.26202	99533504
554.7639	1131081
555.10246	1343129
555.26493	29933582
556.26872	5622094
556.28593	1420051
557.26771	871121
566.26084	1532021
590.2355	1309085
611.28373	796868
617.25632	1446124
639.2406	683797
"""

    from six import StringIO

    sampledata = [StringIO(s()) for s in (_sample1, _sample2, _sample3)]
    
    print('Reading spectra to align ------------')
    samples = [read_spectrum(s) for s in sampledata]
    for sample in samples:
        print(sample,'\n')
        print('------------')

    ppmtol = 1.0
    min_samples = 1
    
    inputs = samples
    
    aligned = align_spectra(inputs, verbose=True)
    print(aligned)
    print('=========================')
    print('\n\nTESTING alignment with aligned inputs')
    
    inputs = samples[:2]
    
    aligned2 = align_spectra(inputs, verbose=True)
    print(aligned2)
    print('-----------------------------------------')
    print('Now the final alignment')
    
    inputs = [aligned2, samples[2]]
    
    aligned_mix = align_spectra(inputs, verbose=True)
    
    print(aligned_mix)
    
    fname = 'data/data_to_align.xlsx'
    out_fname = 'data/aligned_data.xlsx'

    header_row = 2
    sample_names = ['S1', 'S2', 'S3']
    labels = ['wt', 'mod', 'mod']
    sample_names = 1

    aligned = align_spectra_in_excel(fname,
                           save_to_excel=out_fname,
                           ppmtol=ppmtol,
                           min_samples=min_samples,
                           labels=labels,
                           header_row=header_row,
                           sample_names=sample_names)
        
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for sheetname in aligned:
        print(aligned[sheetname])
        print('-----------------------------------')

