"""Similarity of peak lists based on peak assignments (m/z, formulae)."""

import numpy as np
import pandas as pd

def mz_similarity(aligned):
    """Compute counts and Jaccard index by samples."""
    similarities = SimilarityMeasures()
    
    n = len(aligned.sample_names)
    mzs = [aligned.sample(n).mz for n in aligned.sample_names]
    similarities.sample_names = aligned.sample_names[:]
    
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
    similarities.sample_intersection_counts = common_matrix
    similarities.sample_similarity_jaccard = jaccard_matrix
    
    if aligned.labels is not None:
        # build list of unique labels
        slabels = [aligned.labels[0]]
        for label in aligned.labels[1:]:
            if label not in slabels:
                slabels.append(label)
        mzs = [aligned.label(lbl).mz for lbl in slabels]
        # compute intersection counts and Jaccard index
        n = len(slabels)
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
        similarities.label_intersection_counts = common_matrix
        similarities.label_similarity_jaccard = jaccard_matrix
        similarities.unique_labels = slabels
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
        df = pd.DataFrame(self.sample_intersection_counts,
                          columns=self.sample_names,
                          index=self.sample_names)
        res.append(str(df))
        res.append('\nSample similarity, Jaccard indexes')
        df = pd.DataFrame(self.sample_similarity_jaccard,
                          columns=self.sample_names,
                          index=self.sample_names)
        res.append(str(df))
        
            
        if self.label_intersection_counts is not None:
            res.append('\nLabel similarity, counts of common peaks')
            df = pd.DataFrame(self.label_intersection_counts,
                              columns=self.unique_labels,
                              index=self.unique_labels)
            res.append(str(df))
            res.append('\nLabel similarity, Jaccard indexes')
            df = pd.DataFrame(self.label_similarity_jaccard,
                              columns=self.unique_labels,
                              index=self.unique_labels)
            res.append(str(df))
        return "\n".join(res)

