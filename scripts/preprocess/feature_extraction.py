import pickle
import numpy as np
import sys
import glob
import copy
import pdb
import cv2 as cv
import os, os.path
from os.path import join
from scipy import spatial
import matplotlib.pyplot as plt
from itertools import combinations, dropwhile
from collections import Counter
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import color
from sklearn.preprocessing import normalize
from skimage.transform import integral_image
from skimage.feature import local_binary_pattern
from skimage.feature import haar_like_feature
from PIL import Image
import pickle
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.datasets import cifar100, cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16


def preprocess(images):
    shapes = np.array([len(i.shape) for i in images])
    grayImagePos = np.where(shapes==2)[0]
    for i in grayImagePos:
        images[i] = cv.cvtColor(images[i],cv.COLOR_GRAY2BGR)
    images = np.array([cv.resize(i, (224,224), interpolation = cv.INTER_CUBIC) for i in images])
    return images


class FeatureExtractor:
    def __init__(self, mode):
        if mode == 0:
            temp = VGG16(weights="imagenet", pooling="max", include_top=False)
            self.model = Model(temp.input, temp.get_layer('block5_pool').output)
        elif mode == 1:
            self.filters = []
            for scale in [5,10]:
                for freq in [1,2,4,16]:
                    for theta in np.arange(0, np.pi, np.pi / 8):
                        kern = cv.getGaborKernel((5, 5), freq, theta, scale, 0.5, 0, ktype=cv.CV_32F)
                        kern /= 1.5*kern.sum()
                        self.filters.append(kern)
            self.filters = np.array(self.filters)

    @staticmethod # HOG
    def hog(img):
        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(14,14),
                        cells_per_block=(1, 1),block_norm='L2', visualize=True, multichannel=True)
        return fd#, hog_image

    @staticmethod # Sobel
    def sobel(img):
        imageArray = np.zeros(img.shape)
        img = cv.GaussianBlur(img, (5, 5), 0)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        grad_x = cv.Sobel(gray, cv.CV_16S, 1, 0, ksize=3, scale=2, delta=0, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(gray, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        imageArray[grad >= 120] = 1
        return imageArray.ravel()#,grad

    @staticmethod # Color Histogram
    def colorHistogram(img):
        blue = cv.calcHist([img],[0],None,[256],[0,255])
        green = cv.calcHist([img],[1],None,[256],[0,255])
        red = cv.calcHist([img],[2],None,[256],[0,255])
        hist = np.concatenate([blue, green, red], axis=1)
        return hist.ravel()

    @staticmethod # Haar
    def haar(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_ii = integral_image(img)
        haar_2x = haar_like_feature(img_ii, 0, 0, 5, 5, 'type-2-x')
        haar_2y = haar_like_feature(img_ii, 0, 0, 5, 5, 'type-2-y')
        haar_3x = haar_like_feature(img_ii, 0, 0, 5, 5, 'type-3-x')
        haar_3y = haar_like_feature(img_ii, 0, 0, 5, 5, 'type-3-y')
        haar_4  = haar_like_feature(img_ii, 0, 0, 5, 5, 'type-4')
        haar = np.concatenate([haar_2x,haar_2y,haar_3x,haar_3y,haar_4])
        return haar

    @staticmethod # LBP
    def lbp(img):
        radius = 4
        n_points = 16 * radius
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(img, n_points, radius, "uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        return hist#, lbp

    # Gabor
    def gabor(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
        accum = np.zeros_like(img)
        for kern in self.filters:
            fimg = cv.filter2D(img, cv.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum.ravel()

    # VGG-16
    def vgg(self, images):
        images = preprocess_input(images)
        features = self.model.predict(images)
        features = features.reshape(features.shape[0],-1)
        return features

'''
# Extract Features
dataPath = "/mnt/kerem/CBIR/data"
savePath = "/mnt/kerem/CBIR/cache"

datasets = ["oxford", "cifar10"]

gfext = FeatureExtractor(mode=1)
vggfext = FeatureExtractor(mode=0)

methods = [FeatureExtractor.hog, FeatureExtractor.sobel, FeatureExtractor.colorHistogram,
              gfext.gabor, FeatureExtractor.lbp, FeatureExtractor.haar, vggfext.vgg]

for d in datasets:
    minmaxs = {m.__name__:(0,0) for m in methods}
    if d == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = preprocess(x_train)
        x_test = preprocess(x_test)
    else:
        with open(join(dataPath, d+"_images"), "rb") as f:
            images = pickle.load(f)
        with open(join(dataPath, d+"_labels"), "rb") as f:
            labels = pickle.load(f)
        images = preprocess(images)
        x_train,x_test,y_train,y_test = train_test_split(images,labels,test_size=0.2,shuffle=True,stratify=labels,random_state=6)
        with open(join(savePath, d+"_train_labels.pickle"), 'wb') as handle:
            pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(join(savePath, d+"_test_labels.pickle"), 'wb') as handle:
            pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    for f in methods:
        if f.__name__ == "vgg":
            train = f(x_train)
            test = f(x_test)
        else:
            train = np.array([f(img) for img in x_train])
            train_min = np.min(train, axis=0)
            train_max = np.max(train, axis=0)
            minmaxs[f.__name__] = (train_min, train_max)
            train = normalize(train, axis=0, norm='max')
            test = np.array([f(img) for img in x_test])
            test = (test - train_min) / (train_max - train_min + 1e-20)
        with open(join(savePath, d+"_"+str(f.__name__)+"_train.pickle"), 'wb') as handle:
            pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(join(savePath, d+"_"+str(f.__name__)+"_test.pickle"), 'wb') as handle:
            pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Method ", str(f.__name__), " is completed.")
    with open(join(savePath, d+"_minmax.pickle"), 'wb') as handle:
        pickle.dump(minmaxs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Dataset ", d, " is completed.")
'''
