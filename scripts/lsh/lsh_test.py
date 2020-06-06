#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import sys
import cv2 as cv
import os, os.path
from os.path import join
import matplotlib.pyplot as plt
import glob
import copy
from scipy import spatial
from itertools import combinations, dropwhile
import pdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import cifar100, cifar10
from sklearn.preprocessing import normalize
from PIL import Image
import pickle

import lsh.locality_sensitive_hashing as lsh
import lsh.min_hashing as mh
import lsh.euclidean_projection as ep
import lsh.cosine_projection as cos
from preprocess.feature_extraction import FeatureExtractor


mainPath = "/mnt/kerem/CBIR"
featuresPath = "/mnt/kerem/CBIR/features"
dataPath = "/mnt/kerem/CBIR/data"

# # Load Features

datasets = ["oxford", "cifar10"]
methods = ["hog", "gabor", "colorHistogram", "lbp", "haar", "sobel", "vgg", "dae", "pca"]


# ### Load Single Dataset and Extracted Features

d = datasets[1]
# Load Dataset
if d == "oxford":
    with open(join(dataPath, d+"_images"), "rb") as f:
        x = pickle.load(f)
    with open(join(dataPath, d+"_labels"), "rb") as f:
        y = pickle.load(f)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,stratify=y,random_state=6)
else:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

scaler = MinMaxScaler()
m = methods[-3]
# Load Extracted Features
with open(join(featuresPath, d+"_"+m+"_train.pickle"), 'rb') as handle:
    train = pickle.load(handle)
#train = scaler.fit_transform(train)

with open(join(featuresPath, d+"_"+m+"_test.pickle"), 'rb') as handle:
    test = pickle.load(handle)
#test = scaler.transform(train)
print("Number of features: ", train.shape[1])

# # Algorithms

# ### Cosine LSH 

# COSINE PROJECTION
cs = cos.CosineProjection(250, train.shape[1])
cosine_shingles = cs.cosineValues(train)
print("Cosine Projection is done.")

# MIN-HASHING 
min_hash = mh.MinHashing(cosine_shingles, 105, 100)
signature = min_hash.generate_signature()
print("Min Hashing is done.")

# LSH
lsh_ = lsh.LocalitySensitiveHashing(15, 7)
lsh_.fill_hash_tables(signature)
print("LSH is done.")

# Query Image
maps = []
for ind in range(len(y_test)):
    fea_q = test[ind]

    cos_q = cs.cosineValues(np.expand_dims(fea_q, axis=0))
    sig_q = min_hash.generate_signature(cos_q)[0]
    candidates = lsh_.query_candidate(sig_q, min_match=1)
    candidates = sorted(candidates, key=candidates.get, reverse=True)[:10]
    if len(candidates) == 0:
        continue
    count = 0
    for k in candidates:
        #sim = np.sqrt(np.sum((fea_q - train_features[k])**2))
        if y_test[ind] == y_train[k]:    
            count += 1
    if ind % 100 == 0:
        print(ind, "th image is completed.\tMaP: ", np.mean(maps))
    maps.append(count / len(list(candidates)))
print("\nMAP= ", np.mean(maps))


# ### Euclidean LSH

bucket_count = 2
elsh = ep.EuclideanLSH(signature_size=500, n_features=train.shape[1])
elsh.project_points(train, bucket_count=bucket_count)

# Query Image
maps = []
for ind in range(len(y_test)):
    fea_q = test[ind]

    candidates = elsh.query_candidate(fea_q, bucket_count=bucket_count)
    candidates = sorted(candidates, key=candidates.get, reverse=True)[:10]
    count = 0
    for k in candidates:
        #sim = np.sqrt(np.sum((fea_q - train_features[k])**2))
        if y_test[ind] == y_train[k]:    
            count += 1
    if ind % 100 == 0:
        print(ind, "th image is completed.\tMaP: ", np.mean(maps))
    maps.append(count / len(list(candidates)))
print("\nMAP= ", np.mean(maps))


# ### Jaccard LSH 

# ##### Binarize the training and test features if needed

train[train != 0] = 1
train = train.astype(np.int8)
test[test != 0] = 1
test = test.astype(np.int8)

# MIN-HASHING 
min_hash = mh.MinHashing(train, 1050, 100)
signature = min_hash.generate_signature()
print("Min Hashing is done.")


# LOCALITY SENSITIVE HASHING 
lsh_ = lsh.LocalitySensitiveHashing(150, 7)
lsh_.fill_hash_tables(signature)
print("LSH is done.")


# Query Image
maps = []
for i in range(len(y_test)):
    queryFeature = test[i]
    querySignature = min_hash.generate_signature(np.expand_dims(queryFeature, axis=0))[0]
    candidates = lsh_.query_candidate(querySignature, min_match=1)
    candidates = sorted(candidates, key=candidates.get, reverse=True)[:10]
    if len(candidates) == 0:
        continue
    count = 0
    for k in candidates:
        #sim = np.sqrt(np.sum((fea_q - train_features[k].ravel())**2))
        if y_test[i] == y_train[k]:    
            count += 1
    if i % 50 == 0:
        print(i, "th image is completed.\tMaP: ", np.mean(maps))
    maps.append(count / len(list(candidates)))
print("\nMAP= ", np.mean(maps))
