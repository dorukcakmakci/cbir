import pdb 
import sys 
import numpy as np
import matplotlib.pyplot as plt 
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
sys.path.insert(1, "../")
from load_data import oxford_images, oxford_labels, oxford_categories
from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split
from PIL import Image
import pickle

class VGGNet():
    def __init__(self):
        temp = VGG16(weights="imagenet", pooling="max", include_top=False)
        self.model = Model(temp.input, temp.get_layer('block5_pool').output)
    def extract_features(self, image):
        img = Image.fromarray(image)
        img = np.array(img.resize((224, 224), Image.ANTIALIAS))
        if len(img.shape) == 2: # image is grayscale
            img = np.stack((img,img,img), axis=2)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        ftr = self.model.predict(img)
        ftr = ftr[0] / np.linalg.norm(ftr[0])
        return ftr
        

if __name__ == "__main__":
    model = VGGNet()

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

    pdb.set_trace()
    # min max normalize images
    x_train_normalized = x_train_resized.astype('float32') / 255.
    x_test_normalized = x_test_resized.astype('float32') / 255.
    pdb.set_trace()
    
    x_train_preprocessed = preprocess_input(x_train_normalized)
    x_test_preprocessed = preprocess_input(x_test_normalized)

    x_train_features = model.model.predict(x_train_preprocessed)
    x_test_features = model.model.predict(x_test_preprocessed)
    pdb.set_trace()

    np.save("./oxford_train_features.npy", x_train_features)
    np.save("./oxford_test_features.npy", x_test_features)
    np.save("./oxford_train_labels.npy", y_train)
    np.save("./oxford_test_labels.npy", y_test)
    np.save("./oxford_train_images.npy", x_train)
    np.save("./oxford_test_images.npy", x_test)