import pdb
import numpy as np
import sys
import os 
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
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
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))  # adapt this if using `channels_first` image data format

noise_factor = 1

x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=0.1, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=0.1, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


def train_model():
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(8, (4, 4), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (4, 4), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)


    x = Conv2D(32, (4, 4), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (4, 4), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (4, 4), activation='relu', padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (4, 4), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(x_train_noisy, x_train,
                    epochs=100,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))

    autoencoder.save('../../models/cifar10/dae_100epochs.h5')
    pdb.set_trace()

train_model()


