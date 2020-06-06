#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math

class CosineProjection:

    def __init__(self, signature_size, n_features):
        self.ref_planes = np.random.randn(signature_size, n_features)
        self.n_features = n_features
    
    def __signature_bit(self, data):
        sign = np.array([np.dot(data, p) >= 0 for p in self.ref_planes]).astype(np.int8)
        return sign
    
    def cosineValues(self, images):
        signatures = np.array([self.__signature_bit(i) for i in images])
        return signatures
