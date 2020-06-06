#!/usr/bin/env python
# coding: utf-8

import numpy as np
import lsh.prime

class MinHashing:
    
    def __init__(self, shingles, signature_size, k):
        self.shingles = shingles
        self.signature_size = signature_size
        self.k = k
        self.N = self.shingles.shape[1]
        
        # Universal Hashing parameters ((ax+b)%p)%N
        self.a = np.random.randint(self.k, size=self.signature_size)
        self.b = np.random.randint(self.k, size=self.signature_size)
    
    # Generates signature matrix from shingle matrix C
    def generate_signature(self, shingles=np.array([])):
        if shingles.size == 0:
            shingles = self.shingles
        signature = np.full((len(shingles),self.signature_size), np.inf)
        hash_values = self.__get_all_row_hash()
        for i in range(shingles.shape[0]):
            ind = np.argwhere(shingles[i] > 0)
            if ind.size == 0:
                continue
            min_hash_values = np.amin(hash_values[ind], axis=0)  
            signature[i] = min_hash_values
        return signature
    
    # Creates signature_size hash functions and returns the values
    def __universal_hash(self,x):
        c = self.a*x+self.b
        return [(item % self.p)%self.N for item in c]

    # Returns al row-hash values of function h(i)
    def __get_row_hash(self,i):
        result = np.ones((self.N), dtype=int)
        for k in range(self.N):
            result[k] = ((self.a[i]*k+self.b[i])%self.p)%self.N
        return result

    # Returns al row-hash values of function h(i)
    def __get_all_row_hash(self):
        p = lsh.prime.next_prime(self.N)
        result = np.ones((self.N, self.signature_size), dtype=int)
        for k in range(self.N):
            result[k] = ((self.a*(k)+self.b)%p)%self.N
        return result