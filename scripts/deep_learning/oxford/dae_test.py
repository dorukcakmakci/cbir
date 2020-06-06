import pdb
import numpy as np
import sys
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model, load_model
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from load_data import oxford_images, oxford_labels, oxford_categories

# load and split data to training and validation datasets
from tensorflow.keras.datasets import cifar10

# create train and test 
x_train, x_test, y_train, y_test = train_test_split(oxford_images, oxford_labels, test_size=0.2, stratify=oxford_labels, random_state=6)

# resize train dataset images to 224 x 224 x 3
x_train_resized = []
for tr in x_train:
    if len(tr.shape) == 2: # grayscale image
        temp = np.stack((tr,tr,tr), axis=2)
        x_train_resized.append(np.array(Image.fromarray(temp).resize((224, 224), Image.ANTIALIAS)))
    else: # RGB image
        x_train_resized.append(np.array(Image.fromarray(tr).resize((224, 224), Image.ANTIALIAS)))
x_train_resized = np.array(x_train_resized)

# resize test dataset images to 224 x 224 x 3
x_test_resized = []
for tr in x_test:
    if len(tr.shape) == 2: # grayscale image
        temp = np.stack((tr,tr,tr), axis=2)
        x_test_resized.append(np.array(Image.fromarray(temp).resize((224, 224), Image.ANTIALIAS)))
    else: # RGB image
        x_test_resized.append(np.array(Image.fromarray(tr).resize((224, 224), Image.ANTIALIAS)))
x_test_resized = np.array(x_test_resized)

# min max normalize images
x_train_normalized = x_train_resized.astype('float32') / 255.
x_test_normalized = x_test_resized.astype('float32') / 255.

y_train = np.array(y_train)
y_test = np.array(y_test)

# load model
dae = load_model("../../models/oxford/dae_80epochs.h5")
# create layers
input_img = Input(shape=(224, 224, 3))
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
train_features = model.predict(x_train_normalized)
# extract features from test set
test_features = model.predict(x_test_normalized)

# save images, labels and features
np.save("oxford_dae_train_features.npy", train_features)
np.save("oxford_dae_test_features.npy", test_features)
np.save("oxford_dae_train_images.npy", x_train)
np.save("oxford_dae_test_images.npy", x_test)
np.save("oxford_dae_train_labels.npy", y_train)
np.save("oxford_dae_test_labels.npy", y_test)
pdb.set_trace()