"""Similarity of peak lists based on peak assignments (m/z, formulae)."""

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

