#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import Counter, defaultdict
from itertools import combinations, dropwhile

class EuclideanLSH:
    
    def __init__(self, signature_size, n_features):
        self.hash_lines = self.__create_random(count=signature_size, dimension=n_features)
        self.signature_size = signature_size
        self.n_features = n_features
        self.clear_hash_tables()

    def query_candidate(self, point, bucket_count):
        similars = []
        for j in range(self.signature_size):
            norm = np.dot(point, self.hash_lines[j])
            signs = self.__sign(self.hash_lines[j][0]) * self.__sign(norm)
            hash_val = ((np.multiply(norm,signs) + 1) / bucket_count).astype(int)
            if hash_val in self.hash_tables[j]:
                similars += self.hash_tables[j][hash_val]
        similars = Counter(similars)
        return similars
    
    def project_points(self, x, bucket_count):
        # Apply each hash function
        for j in range(self.signature_size):
            norm = np.dot(x, self.hash_lines[j])
            signs = self.__sign(self.hash_lines[j][0]) * self.__sign(norm)
            hash_vals = ((np.multiply(norm,signs) + 1) / bucket_count).astype(int)
            for m,h in enumerate(hash_vals):
                self.hash_tables[j][h].append(m)            
                
    def generate_candidates(self, min_match=None):
        candidate_pairs = []
        for table in self.hash_tables:
            for row in table:
                bucket = table[row]
                pairs = combinations(bucket, 2)
                for elements in pairs:
                    candidate_pairs.append(tuple(sorted(elements)))
        if not min_match:
            min_match = len(self.hash_tables)/20
        # Take most occuring pairs, not all of them
        result = []
        for e in candidate_pairs:
            count = candidate_pairs.count(e)
            if count > min_match:
                result.append(e)
        return np.array(set(result))
    
    def clear_hash_tables(self):
        self.hash_tables = []
        for i in range(self.signature_size):
            self.hash_tables.append(defaultdict(list))

    def __sign(self, x):
        return ((x >= 0)*2)-1

    def __create_random(self, count, dimension):
        vectors = np.zeros((count,dimension))
        for i in range(count):
            v = (2*np.random.rand(dimension))-1
            vectors[i] = v / (v**2).sum()**0.5
        return vectors
    
    def __filterOut(self, candidates, min_count):
        for key, count in dropwhile(lambda key_count: key_count[1] >= min_count, candidates.most_common()):
            del candidates[key]
        return candidates
    