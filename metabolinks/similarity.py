"""Similarity of peak lists based on peak assignments (m/z, formulae)."""

from itertools import chain, combinations

import numpy as np
import pandas as pd

import metabolinks.dataio as dataio
import metabolinks.datasets as datasets

def mz_similarity(dataset, has_labels=False):
    """Compute counts and Jaccard index by samples."""
    if has_labels:
        acc = dataset.ms
    else:
        acc = dataset.ums
    
    similarities = SimilarityMeasures()
    sample_names = list(acc.unique_samples)

    n = len(sample_names)
    mzs = [acc.features(sample=name) for name in sample_names]
    similarities.sample_names = sample_names[:]
    
    common_matrix = np.zeros((n, n), dtype=int)
    jaccard_matrix = np.zeros((n,n))
    
    for i in range(n):
        common_matrix[i, i] = len(mzs[i])
        jaccard_matrix[i, i] = 1.0
    
    for i1 in range(n-1):
        for i2 in range(i1+1, n):
            mz1 = mzs[i1]
            mz2 = mzs[i2]
            set1 = set(mz1)
            set2 = set(mz2)
            u12 = set1.union(set2)
            i12 = set1.intersection(set2)
            ni12 = len(i12)
            common_matrix[i1, i2] = ni12
            common_matrix[i2, i1] = ni12
            jaccard = ni12 / len(u12)
            jaccard_matrix[i2, i1] = jaccard
            jaccard_matrix[i1, i2] = jaccard

    similarities.sample_intersection_counts = pd.DataFrame(common_matrix, columns=sample_names, index=sample_names)
    similarities.sample_similarity_jaccard = pd.DataFrame(jaccard_matrix, columns=sample_names, index=sample_names)
    
    if has_labels:
        labels = acc.unique_labels

        mzs = [acc.features(label=lbl) for lbl in labels]
        # compute intersection counts and Jaccard index
        n = len(labels)
        common_matrix = np.zeros((n, n), dtype=int)
        jaccard_matrix = np.zeros((n,n))
        
        for i in range(n):
            common_matrix[i, i] = len(mzs[i])
            jaccard_matrix[i, i] = 1.0
        
        for i1 in range(n-1):
            for i2 in range(i1+1, n):
                mz1 = mzs[i1]
                mz2 = mzs[i2]
                set1 = set(mz1)
                set2 = set(mz2)
                u12 = set1.union(set2)
                i12 = set1.intersection(set2)
                ni12 = len(i12)
                common_matrix[i1, i2] = ni12
                common_matrix[i2, i1] = ni12
                jaccard = ni12 / len(u12)
                jaccard_matrix[i2, i1] = jaccard
                jaccard_matrix[i1, i2] = jaccard
        similarities.label_intersection_counts = pd.DataFrame(common_matrix, columns=labels, index=labels)
        similarities.label_similarity_jaccard = pd.DataFrame(jaccard_matrix, columns=labels, index=labels)
        similarities.unique_labels = labels
    return similarities

class SimilarityMeasures(object):
    """A container that holds the results of similarity measures."""
    
    def __init__(self):
        self.sample_intersection_counts = None
        self.sample_similarity_jaccard = None
        self.label_intersection_counts = None
        self.label_similarity_jaccard = None
        self.unique_labels = None
        self.sample_names = None
    
    def __str__(self):
        res = ['\nSample similarity, counts of common peaks']
        res.append(str(self.sample_intersection_counts))
        res.append('\nSample similarity, Jaccard indexes')
        res.append(str(self.sample_similarity_jaccard))

        if self.label_intersection_counts is not None:
            res.append('\nLabel similarity, counts of common peaks')
            res.append(str(self.label_intersection_counts))
            res.append('\nLabel similarity, Jaccard indexes')
            res.append(str(self.label_similarity_jaccard))
        return "\n".join(res)

def common(objects):
    """Given a list `objects` of data objects, compute common features (intersection).
    
       Accepts a sequence of Index, Series or DataFrame.
       Returns an Index with common features"""
    
    # ensure a list of Indexes
    if not objects:
        return pd.Index([])
    if hasattr(objects[0], 'index'):
        objects = [s.index for s in objects]
    # Calculate intersection
    index = objects[0]
    for right in objects[1:]:
        index = index.intersection(right)
    return index

def exclusive(objects):
    """Given a list `objects` of data objects, compute exclusive features for each object.
    
       Accepts a sequence of Index, Series or DataFrame.
       Returns a list of Indexes with exclusive features for each object."""
    
    # ensure a list of Indexes
    if not objects:
        return pd.Index([])
    if hasattr(objects[0], 'index'):
        objects = [s.index for s in objects]

    # concat all indexes
    concatenation = objects[0]
    for right in objects[1:]:
        concatenation = concatenation.append(right)
    
    # find indexes that occur only once
    reps = concatenation.value_counts()
    exclusive_feature_counts = reps[reps == 1]
    
    # keep only those in each sample

    exclusive = [s[s.isin(exclusive_feature_counts.index)] for s in objects]
    return exclusive

def compute_Venn_features(objects, names, max_intersections=None):
    """Computes a dict {tuple: index} suitable to fill Venn diagrams of common features.

       Names of objects are used in the tuples (dict keys). A single name means exclusive features
       more than two names means features for the given slot in the Venn diagram.
       dict values are Indexes, representing features: their len() will provide counts,
       but they can be used to index DataFrames to obtain data.
       
       Since the returned dict results from a 'power set' enumeration, (size 2**n -1), argument
       `max_intersections` can be used to restrict the size of the result.
       For example, if `max_intersections` == 2, then only exclusive or pair
       intersections are computed."""
       
    if max_intersections is None:
        max_intersections = len(objects)
    
    # ensure a list of Indexes
    if not objects:
        return {}
    if hasattr(objects[0], 'index'):
        objects = [obj.index for obj in objects]

    # concat all indexes
    concatenation = objects[0]
    for right in objects[1:]:
        concatenation = concatenation.append(right)
    
    # find ocorrence counts of features
    reps = concatenation.value_counts()
    
    # group features by number of ocorrences
    count_groups = [reps[reps == (i+1)].index for i in range(max_intersections)]
    # for i, c in enumerate(count_groups):
    #     print('----------------')
    #     print(i+1, '--->', list(c))
    #     print('----------------')

    # enumerate power set
    res = {}

    s = list(range(len(objects)))
    for t in chain.from_iterable(combinations(s, r) for r in range(len(s)+1)):
        if len(t) > max_intersections:
            break
        elif len(t) == 0:
            # empty subset, skip
            continue
        else:
            # common features with exactely len(t) ocorrences
            if len(t) == 1:
                all_feats = objects[t[0]]
            else:
                all_feats = common([objects[i] for i in t])
            features = all_feats[all_feats.isin(count_groups[len(t)-1])]
            subset_names = tuple([names[i] for i in t])
            res[subset_names] = features
    return res

if __name__ == "__main__":
    import six
    print('Reading from string data (as io stream) with labels------------\n')
    dataset = dataio.read_data_csv(six.StringIO(datasets.demo_data2()), has_labels=True)
    print(dataset)
    print('-- info --------------')
    print(dataset.ms.info())
    print('-- global info---------')
    print(dataset.ms.info(all_data=True))
    print('-----------------------')
    print('***** SIMILARITY MEASURES ****')
    similarities = mz_similarity(dataset, has_labels=True)
    print(similarities)

    print('***** FEATURE OVERLAP (AND VENN DIAGRAM CALCULATIONS ****')
    print('--- example data sets')
    s1 = pd.DataFrame({'Bucket label': ['A0', 'A1', 'A2', 'A3'],
                        'Name': ['B0', np.nan, 'B2', 'B3'],
                        'Formula': ['C0', 'C1', 'C2', 'C3']},
                        index=[0, 1, 2, 3]).set_index('Bucket label')
    s2 = pd.DataFrame({'Bucket label': ['A0', 'A1', 'A2', 'A4'],
                        'Name': ['B0', np.nan, 'B2', 'B4'],
                        'Formula': ['C0', 'C1', 'C2', 'C4']},
                        index=[0, 1, 2, 3]).set_index('Bucket label')

    s3 = pd.DataFrame({'Bucket label': ['A0', 'A1', 'A4', 'A7'],
                        'Name': ['B0', np.nan, 'B4', 'B7'],
                        'Formula': ['C0', 'C1', 'C4', 'C7']},
                        index=[0, 1, 2, 3]).set_index('Bucket label')
    print(s1, end='\n------------------------\n')
    print(s2, end='\n------------------------\n')
    print(s3)

    samples = [s1, s2, s3]

    print('\n---------- Common to all')
    common_to_all = common(samples)
    print(list(common_to_all))

    print('\n---------- Common features for every combination of two samples')
    pair_intersections = {}
    n = len(samples)
    for i in range(n-1):
        for j in range(i+1, n):
            pair_intersections[(i,j)] = common([samples[i], samples[j]])

    for (i, j) in pair_intersections:
        print(f'\n--- common between s{i} and s{j}')
        print(list(pair_intersections[(i,j)]))


    print('\n----------- Exclusive features')
    exclusive_features = exclusive(samples)

    for i, e in enumerate(exclusive_features):
        print('\n---- Exclusive to sample', i+1)
        print(list(e))

    print('\n----------- Features for Venn diagram')

    venn = compute_Venn_features(samples, 'S1 S2 S3'.split())
    for t in venn:
        print(f'{str(t):>20} --> {list(venn[t])}')
