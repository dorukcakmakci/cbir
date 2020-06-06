#!/usr/bin/env python
# coding: utf-8

import numpy as np
from itertools import combinations, dropwhile
from collections import Counter, defaultdict

class LocalitySensitiveHashing:

    def __init__(self, B, R):
        self.B = B
        self.R = R
        self.clear_hash_tables()
        
    def query_candidate(self, q_signature, min_match):
        similars = []
        for b in range(self.B):
            key = self.__get_number(q_signature[b*self.R:(b+1)*self.R])
            if key in self.hash_tables[b]:
                similars += self.hash_tables[b][key]
        similars = Counter(similars)
        return similars
                         
    def generate_candidates(self):
        candidate_pairs = []
        for b in range(self.B):
            for row in self.hash_tables[b]:
                bucket = self.hash_tables[b][row]
                pairs = combinations(bucket, 2)
                for elements in pairs:
                    candidate_pairs.append(tuple(sorted(elements)))
        return set(candidate_pairs)
    
    def fill_hash_tables(self, signature):
        for b in range(self.B):
            for i in range(len(signature)):
                key = self.__get_number(signature[i,b*self.R:(b+1)*self.R])
                self.hash_tables[b][key].append(i)
    
    def clear_hash_tables(self):
        self.hash_tables = []
        for i in range(self.B):
            self.hash_tables.append(defaultdict(list))
    
    @staticmethod
    def __get_number(array):
        number = 0
        base = 2
        decimal = base**(len(array)-1)
        for i in array:
            number += decimal * i
            decimal /= base
        if number == np.inf or number == -np.inf:
            return -1
        else:
            return int(number)
        
    def __filterOut(self, candidates, min_count):
        for key, count in dropwhile(lambda key_count: key_count[1] >= min_count, candidates.most_common()):
            del candidates[key]
        return candidates
  