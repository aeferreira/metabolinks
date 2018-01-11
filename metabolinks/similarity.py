"""Similarity of peak lists based on peak assignments (m/z, formulae).
"""

import numpy as np

class SimilarityMeasures(object):
    """A container that holds the results of similarity measures.
    """
    
    def __init__(self):
        self.sample_similarity = None
        self.label_similarity = None
        self.unique_labels = None


def compute_similarity_measures(aligned):
    # compute counts and Jaccard index by samples
    similarities = SimilarityMeasures()
    
    n = len(aligned.sample_names)
    smatrix = np.zeros((n, n))
    for i1 in range(n-1):
        for i2 in range(i1+1, n):
            mz1 = aligned.sample(aligned.sample_names[i1]).mz
            mz2 = aligned.sample(aligned.sample_names[i2]).mz
            smatrix[i1, i1] = len(mz1)
            smatrix[i2, i2] = len(mz2)
            set1 = set(mz1)
            set2 = set(mz2)
            u12 = set1.union(set2)
            i12 = set1.intersection(set2)
            smatrix[i1, i2] = len(i12)
            jaccard = len(i12) / len(u12)
            smatrix[i2, i1] = jaccard
    similarities.sample_similarity = smatrix
    
    if aligned.labels is not None:
        # build list of unique labels
        slabels = [aligned.labels[0]]
        for i in range(1, len(aligned.labels)):
            label = aligned.labels[i]
            if label not in slabels:
                slabels.append(label)
        mzs = {}
        for label in slabels:
            mzs[label] = aligned.label(label).mz
        # compute intersection counts and Jaccard index
        n = len(slabels)
        lmatrix = np.zeros((n, n))
        for i1 in range(n-1):
            for i2 in range(i1+1, n):
                label1 = slabels[i1]
                label2 = slabels[i2]
                set1 = set(mzs[label1])
                set2 = set(mzs[label2])
                lmatrix[i1, i1] = len(set1)
                lmatrix[i2, i2] = len(set2)
                u12 = set1.union(set2)
                i12 = set1.intersection(set2)
                lmatrix[i1, i2] = len(i12)
                jaccard = len(i12) / len(u12)
                lmatrix[i2, i1] = jaccard
        similarities.label_similarity = lmatrix
        similarities.unique_labels = slabels
    return similarities

