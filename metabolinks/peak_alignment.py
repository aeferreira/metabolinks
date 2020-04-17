from __future__ import print_function

import time
import itertools
from collections import OrderedDict

import numpy as np
import pandas as pd

from metabolinks.utils import s2HMS

def group_peaks_hca_median(df, ppmtol=1.0):
    reltol = ppmtol * 1.0e-6

    # data for each cluster
    samples = []
    dist2next = []
    masses = []
    centroids = []

    # fill list with original peaks
    for i in range(len(df)):
        samples.append(set([df.loc[i, '_sample']]))
        masses.append([df.loc[i, 'm/z']])
        centroids.append(df.loc[i, 'm/z'])

    # compute dist2next
    for i in range(len(samples)-1):
        intersection = samples[i].intersection(samples[i+1])
        reldist = (centroids[i+1] - centroids[i]) / centroids[i]
        if len(intersection) > 0 or reldist > reltol:
            dist2next.append(np.inf)
        else:
            dist2next.append(reldist)
    
    dist2next = np.array(dist2next)
    nfinite = np.sum(np.isfinite(dist2next))

    while nfinite > 0:

        # locate minimum
        i = int(dist2next.argmin()) # expected to be the most expensive line
        # print(' ********** i=', i, 'nfinite = ', nfinite, '**** merging ******')
        # print(samples[i])
        # print(masses[i])
        # print('----- and -------')
        # print(samples[i+1])
        # print(masses[i+1])
        
        # merge cluster imin and imin + 1
        newsamples = samples[i].union(samples[i+1])
        newmasses = masses[i] + masses[i+1]
        newcentroid = np.median(newmasses)

        # change arrays
        centroids[i] = newcentroid
        del centroids[i+1]
        masses[i] = newmasses
        del masses[i+1]
        samples[i] = newsamples
        del samples[i+1]

        # recompute distances
        if i != len(dist2next) - 1:
            dist2next = np.delete(dist2next, i+1)
            if i > 0:
                intersection = samples[i-1].intersection(samples[i])
                reldist = (centroids[i] - centroids[i-1]) / centroids[i-1]
                if len(intersection) > 0 or reldist > reltol:
                    dist2next[i-1] = np.inf
                else:
                    dist2next[i-1] = reldist
            if i < len(samples) - 1:
                intersection = samples[i].intersection(samples[i+1])
                reldist = (centroids[i+1] - centroids[i]) / centroids[i]
                if len(intersection) > 0 or reldist > reltol:
                    dist2next[i] = np.inf

                else:
                    dist2next[i] = reldist
        else:
            dist2next = np.delete(dist2next, i)
            if i > 0:
                intersection = samples[i-1].intersection(samples[i])
                reldist = (centroids[i] - centroids[i-1]) / centroids[i-1]
                if len(intersection) > 0 or reldist > reltol:
                    dist2next[i-1] = np.inf
                else:
                    dist2next[i-1] = reldist

        nfinite = np.sum(np.isfinite(dist2next))

    glabels = []
    for glabel, group in enumerate(samples):
        glabels.extend([glabel]*len(group))

    # insert new column with group indexes and group by this column
    result = pd.concat([df, pd.Series(glabels, index=df.index, name='_group')],
                        axis=1)

    return result.groupby('_group')


def group_peaks_naive_1pass_centroid(df, ppmtol=1.0):
    """Compute groups of peaks, naive, one-pass, control distance to centroid ."""
    
    # compute group indexes
    glabel = 0
    start = 0
    glabels = [0]

    reltol = ppmtol * 1.0e-6

    m1 = df.loc[0, 'm/z']
    samples = [df.loc[0, '_sample']]
    masses = [m1]
    centroid = m1

    for i in range(len(df)):
        if i == start:
            continue

        m2 = df.loc[i, 'm/z']
        sample = df.loc[i, '_sample']

        # if i < 20:
        #     print(' ********** i=', i, 'start=', start, 'samples=', samples)
        #     print('d start = ', m1, 'sample', df.loc[start, '_sample'])
        #     print('d i     = ', m2, 'sample', df.loc[i, '_sample'])
        
        if (sample in samples) or ((m2 - centroid) / centroid > reltol):
            # new group
            # if i < 20:
            #     if sample in samples:
            #         print(sample, 'in', samples)
            #     else:
            #         print('d (ppm) =', 1e6 * ((m2 - m1) / m1))
            glabel += 1
            start = i
            m1 = df.loc[start, 'm/z']
            samples = [df.loc[start, '_sample']]
            centroid = m1
            masses=[m1]
        else:
            samples.append(sample)
            masses.append(m2)
            centroid = np.mean(masses)

        glabels.append(glabel)

    # insert new column with group indexes and group by this column
    result = pd.concat([df, pd.Series(glabels, index=df.index, name='_group')],
                        axis=1)

    return result.groupby('_group')

def group_peaks_naive_1pass_complete_linkage(df, ppmtol=1.0):
    """Compute groups of peaks, according to m/z proximity."""
    
    # compute group indexes
    glabel = 0
    start = 0
    glabels = [0]

    reltol = ppmtol * 1.0e-6

    m1 = df.loc[0, 'm/z']
    samples = [df.loc[0, '_sample']]

    for i in range(len(df)):
        if i == start:
            continue

        m2 = df.loc[i, 'm/z']
        sample = df.loc[i, '_sample']

        # if i < 20:
        #     print(' ********** i=', i, 'start=', start, 'samples=', samples)
        #     print('d start = ', m1, 'sample', df.loc[start, '_sample'])
        #     print('d i     = ', m2, 'sample', df.loc[i, '_sample'])

        if (sample in samples) or ((m2 - m1) / m1 > reltol):
            # new group
            # if i < 20:
            #     if sample in samples:
            #         print(sample, 'in', samples)
            #     else:
            #         print('d (ppm) =', 1e6 * ((m2 - m1) / m1))
            glabel += 1
            start = i
            m1 = df.loc[start, 'm/z']
            samples = [df.loc[start, '_sample']]
        else:
            samples.append(sample)

        glabels.append(glabel)

    # insert new column with group indexes and group by this column
    result = pd.concat([df, pd.Series(glabels, index=df.index, name='_group')],
                        axis=1)

    return result.groupby('_group')


def align(inputs, ppmtol=1.0, min_samples=1,
          grouping='hc', # other are 'complete' and 'centroid'
          return_alignment_desc=False,
          verbose=True):

    """Join tables according to feature proximity, interpreting features as m/z values.
    
       m/z values should be contained in the (row) index of the input DataFrames.
       Returns a Pandas DataFrame by outer joining input frames on proximity groups."""

    # get the grouping method as a function
    if grouping == 'hc':
        gfunc = group_peaks_hca_median
    elif grouping == 'complete':
        gfunc = group_peaks_naive_1pass_complete_linkage
    elif grouping == 'centroid':
        gfunc = group_peaks_naive_1pass_centroid
    else:
        raise ValueError(f'Unrecognizable grouping method: "{grouping}"')

    start_time = time.time()

    if verbose:
        print ('------ Aligning tables -------------')
        print (' Samples to align:', [list(sample.columns) for sample in inputs])

        print('- Extracting all features...')

    # Joining data (vertically)
    
    n = len(inputs)
    dfs = []
    
    # tag with sample names, concat vertically and sort by m/z
    for i, sample in enumerate(inputs):
        # create DataFrame with columns 'm/z', '_peak_index', and '_spectrum'
        dfs.append(
            pd.DataFrame({
                'm/z': list(sample.index),
                '_sample': i,
                '_feature': range(len(sample))
            })
        )

    # concat vertically
    cdf = pd.concat(dfs)
    # sort by m/z
    cdf = cdf.sort_values(by='m/z')
    # insert a range index
    cdf.index = list(range(len(cdf)))
    cdf = cdf.astype({'_sample': int, '_feature': int})

    if verbose:
        print('  Done, (total {} features in {} samples)'.format(cdf.shape[0], n))

    # Grouping data and building resulting table

    if verbose:
        print('- Grouping and joining...')

    grouped = gfunc(cdf, ppmtol=ppmtol)

    # print('*********************************')
    # for i, g in zip(range(20), grouped):
    #     print('++++++++++++++++++++++++++++++')
    #     print(g[0])
    #     print('++++++++++++++++++++++++++++++')
    #     print(g[1])
    # print('*********************************')

    # create new indexes for input DataFrames
    new_indexes = [[0]*len(sample) for sample in inputs]

    new_features = []
    mz_range_array = []
    group_labels = []
    group_nfeatures = []
    
    for (group_label, group) in grouped:
        group_labels.append(group_label)
        group_nfeatures.append(len(group))
        new_features.append(group['m/z'].median())
        # new_features.append(group['m/z'].mean())
        m_min, m_max = group['m/z'].min(), group['m/z'].max()
        range_ppm = 1e6 * (m_max - m_min) / m_min
        mz_range_array.append(range_ppm)

        # populate new_indexes with loc information
        for _, row in group.iterrows():
            new_indexes[int(row['_sample'])][int(row['_feature'])] = int(row['_group'])

    # copy over inputs with row indexes indicating groups numbers
    new_inputs = [pd.DataFrame(sample.values,
                               index = new_indexes[i],
                               columns=sample.columns)
                               for i, sample in enumerate(inputs)]
    # print('*********************************')
    # for i, ni in enumerate(new_inputs):
    #     print('++++++++++++++++++++++++++++++')
    #     print(i)
    #     print('++++++++++++++++++++++++++++++')
    #     print(new_indexes[i])
    #     print('++++++++++++++++++++++++++++++')
    #     print(inputs[i])
    #     print('++++++++++++++++++++++++++++++')
    #     print(ni)
    # print('*********************************')

    # perform the join of DataFrames based on indexes containing group numbers
    result = new_inputs[0]
    result = result.join(new_inputs[1:], how='outer')
    result.index = new_features

    alignment_desc = pd.DataFrame({'# features': group_nfeatures,
                                   'mean m/z': new_features,
                                   'm/z range (ppm)': mz_range_array},
                                   index=group_labels)
    
    # Discard rows according to min_samples cutoff
    n_non_discarded = len(result)

    if min_samples > 1:
        where_keep = (alignment_desc['# features'] >= min_samples).values
        result = result[where_keep]
        alignment_desc = alignment_desc[where_keep]

    if verbose:
        print('  Done, {} groups found'.format(n_non_discarded))
        print('Elapsed time: {}\n'.format(s2HMS(time.time() - start_time)))
        # print(result.info())

        if min_samples > 1:
            n_discarded = n_non_discarded - len(result)
            msg = '- {} groups were discarded (#samples < {})'.format
            print(msg(n_discarded, min_samples))

        print(alignment_summary(alignment_desc, ppmtol))

    if return_alignment_desc:
        return (result, alignment_desc)
    else:
        return result

# Alias
align_spectra = align

def alignment_summary(alignment_desc, ppmtol):
        res = []
        lines=['Sample coverage of features']
        cov_items = alignment_desc['# features'].value_counts().sort_index().items()
        for n, c in cov_items:
            lines.append('{:5d} features in {} samples'.format(c, n))
        res.append('\n'.join(lines))
        lines = ['m/z range (ppm) distribution']
        range_hist = np.histogram(alignment_desc['m/z range (ppm)'].values, bins=10, range=(0.0, ppmtol))
        bins = range_hist[1]
        counts = range_hist[0]
        for i, c in enumerate(counts):
            lines.append('  [{:3.1f},{:3.1f}[ : {}'.format(bins[i], bins[i+1], c))
        res.append('\n'.join(lines))
        hist_high = bins[-1]
        excess_ranges = alignment_desc[alignment_desc['m/z range (ppm)'] > hist_high]
        n_excess = len(excess_ranges)
        if n_excess > 0:
            res.append('Peaks with m/z range in excess of tolerance')
            res.append(str(excess_ranges))
        else:
            res.append('  > {:<7.1f} : {}'.format(hist_high, n_excess))
        return '\n'.join(res)


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
    from metabolinks import add_labels, read_data_csv, read_data_from_xcel
    #from metabolinks.dataio import read_data_csv, read_data_from_xcel

    sampledata = [StringIO(s()) for s in (_sample1, _sample2, _sample3)]

    print('Reading spectra to align ------------')
    samples = [read_data_csv(s) for s in sampledata]
    for sample in samples:
        print(sample,'\n')
        print('------------')

    aligned, desc = align_spectra(samples, return_alignment_desc=True, verbose=True)
    print('\n--- Result: --------------------')
    print(aligned)
    print('\n--- groups: --------------------')
    print(desc)
    print('=========================')
    print('\n\nTESTING alignment with aligned inputs')

    aligned2, desc = align_spectra(samples[:2], return_alignment_desc=True, verbose=True)
    print('\n--- Result: --------------------')
    print(aligned2)
    print('\n--- groups: --------------------')
    print(desc)
    print('-----------------------------------------')
    print('Now the final alignment')

    aligned_mix, desc = align_spectra([aligned2, samples[2]], return_alignment_desc=True, verbose=True)
    print('\n--- Result: --------------------')
    print(aligned_mix)
    print('\n--- groups: --------------------')
    print(desc)

    print('=========================')
    print('\n\nTESTING alignment with min_samples cutoff')
    inputs = samples

    aligned, desc = align_spectra(samples, min_samples=2, return_alignment_desc=True, verbose=True)
    print('\n--- Result: --------------------')
    print(aligned)
    print('\n--- groups: --------------------')
    print(desc)

    print('=========================')
    print('\n\nTESTING alignment with several input sets from an Excel file')

    labels = ['wt', 'mod', 'mod']

    # Reading from Excel ----------
    file_name = 'sample_data.xlsx'
    out_fname = 'aligned_data.xlsx'
    import os

    _THIS_DIR, _ = os.path.split(os.path.abspath(__file__))
    fname = os.path.join(_THIS_DIR, 'data', file_name)

    data_sets = read_data_from_xcel(fname, header=[0, 1], drop_header_levels=1)

    print('------ Aligning tables in each Excel sheet...')
    results_sheets = {}
    for d in data_sets:
        print('\n++++++++++++++', d)
        aligned, desc = align(data_sets[d],
                              min_samples=1, 
                              ppmtol=1.0,
                              return_alignment_desc=True,
                              verbose=True)
        aligned = add_labels(aligned, labels)
        aligned.columns.names = ['label', 'sample']
        # aligned.cdl.samples = sample_names
        print('\n--- Result: --------------------')
        print(aligned)
        results_sheets[d] = aligned
        results_sheets['groups {}'.format(d)] = desc
        print('+++++++++++++++++++++++++++++')

    ofname = os.path.join(_THIS_DIR, 'data', out_fname)

    with pd.ExcelWriter(ofname) as writer:
        for sname in results_sheets:
            results_sheets[sname].to_excel(writer, sheet_name=sname)
