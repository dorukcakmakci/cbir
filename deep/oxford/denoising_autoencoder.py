import pdb
import numpy as np
import sys
import os 
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.callbacks import TensorBoard
sys.path.insert(1, "../")
from load_data import oxford_images, oxford_labels, oxford_categories
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# load and split data to training and validation datasets
from tensorflow.keras.datasets import cifar100

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

noise_factor = 1

x_train_noisy = x_train_normalized + noise_factor * np.random.normal(loc=0.0, scale=0.1, size=x_train_normalized.shape)
x_test_noisy = x_test_normalized + noise_factor * np.random.normal(loc=0.0, scale=0.1, size=x_test_normalized.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)



def train_model():
    input_img = Input(shape=(224, 224, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)


    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(x_train_noisy, x_train_normalized,
                    epochs=100,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test_normalized))

    autoencoder.save('../../models/oxford/denoising_autoencoder_larger_100epochs.h5')

train_model()


