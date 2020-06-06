import numpy as np
import os
import pdb
import pickle

# load oxford data 
with open("../../data/data/oxford_images", "rb") as f:
    oxford_images = pickle.load(f)
with open("../../data/data/oxford_labels", "rb") as f:
    oxford_labels = pickle.load(f)
with open("../../data/data/oxford_categories", "rb") as f:
    oxford_categories = pickle.load(f)