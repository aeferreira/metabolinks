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

    # print('------------ control  cdf----------------')
    # print(cdf.head(40))
    # print('--------------------------------------------')

    if verbose:
        print(f'  Done, (total {cdf.shape[0]} features in {n} samples)')

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

    # perform the join of DataFrames based on indexes
    # the indexes contain group numbers
    result = new_inputs[0]
    for i in range(1, len(new_inputs)):
        result = result.join(new_inputs[i], how='outer')
        # print(f'************************ RESULT of joining {i+1} ***************')
        # print(result.head(20))
        # print('*****************************************************************')
    # result = result.join(new_inputs[1:], how='outer')
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

    from metabolinks import add_labels, read_data_csv, read_data_from_xcel
    from metabolinks.datasets import get_data_path

    # get peak-list files locations
    data_folder = get_data_path()
    sample_files = 'peak_list1.txt', 'peak_list2.txt', 'peak_list3.txt'
    sample_files = [data_folder / name for name in sample_files]

    # read files as TSVs
    print('Reading peak lists to align ------------')
    samples = [read_data_csv(s) for s in sample_files]
    for i, sample in enumerate(samples):
        print(sample,'\n')
        print('------------')

    print('\n\n===============================================')
    print('TESTING alignment of 3 peak lists')
    aligned, desc = align_spectra(samples, return_alignment_desc=True, verbose=True)
    print('\n--- Result: --------------------')
    print(aligned.head(30))
    # if saving to file is needed...
    # aligned.to_csv('exp_aligned.csv', sep='\t')
    print('\n--- groups: --------------------')
    print(desc)

    print('\n\n===============================================')
    print('TESTING alignment with 2 already aligned inputs with the third')

    aligned2 = align_spectra(samples[:2], verbose=False)
    print('\n--- Result with 2 first samples: --------------------')
    print(aligned2.head(30))
 

    print('\n------ Now the final alignment...')

    aligned_mix, desc = align_spectra([aligned2, samples[2]], return_alignment_desc=True, verbose=True)
    print('\n\n--- Result: --------------------')
    print(aligned_mix.head(30))
    print('\n--- groups: --------------------')
    print(desc)

    print('\n\n======================================================')
    print('\nTESTING alignment of 3 peak lists with min_samples cutoff')
    inputs = samples

    aligned, desc = align_spectra(samples, min_samples=2, return_alignment_desc=True, verbose=True)
    print('\n--- Result: --------------------')
    print(aligned.head(30))
    print('\n--- groups: --------------------')
    print(desc)

    print('\n\n======================================================')
    print('\n\nTESTING alignment with several input sets from tabs of an Excel file')

    labels = ['wt', 'mod', 'mod']

    file_name = 'sample_data.xlsx'
    out_fname = 'aligned_data.xlsx'

    fname = data_folder / file_name

    # Reading from Excel ----------
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
        print('\n--- Result: --------------------')
        print(aligned.head(30))
        results_sheets[d] = aligned
        results_sheets['groups {}'.format(d)] = desc
        print('+++++++++++++++++++++++++++++')

    ofname = data_folder / out_fname

    print(f'\n------ Saving results in Excel file {ofname.name}...')
    with pd.ExcelWriter(ofname) as writer:
        for sname in results_sheets:
            results_sheets[sname].to_excel(writer, sheet_name=sname)
    print('\n Done!')
