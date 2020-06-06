import pdb
import numpy as np
import sys
import os 
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model, load_model
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# load and split data to training and validation datasets
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3)) 

# load model
dae = load_model("../../models/cifar10/dae_100epochs.h5")
# create layers
input_img = Input(shape=(32, 32, 3))
encoder_layer_1 = dae.layers[1]
encoder_layer_2 = dae.layers[2]
encoder_layer_3 = dae.layers[3]
encoder_layer_4 = dae.layers[4]
encoder_layer_5 = dae.layers[5]
encoder_layer_6 = dae.layers[6]
# forward pass
encoded = encoder_layer_1(input_img)
encoded = encoder_layer_2(encoded)
encoded = encoder_layer_3(encoded)
encoded = encoder_layer_4(encoded)
encoded = encoder_layer_5(encoded)
encoded = encoder_layer_6(encoded)
model = Model(input_img, encoded)
model.summary()
# extract features from train set 
train_features = model.predict(x_train)
# extract features from test set
test_features = model.predict(x_test)
np.save("cifar_dae_train_features.npy", train_features)
np.save("cifar_dae_test_features.npy", test_features)
pdb.set_trace()