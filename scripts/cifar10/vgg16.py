import pdb 
import sys 
import numpy as np
import matplotlib.pyplot as plt 
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from tensorflow.keras.datasets import cifar10
from tqdm import tqdm
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
    temp = VGG16(weights="imagenet", pooling="max", include_top=False)
    model = Model(temp.input, temp.get_layer('block5_pool').output)

    # # # cifar10
    # # (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # # x_train = np.reshape(x_train, (len(x_train), 32, 32, 3)) 
    # # x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

    # # # resize images 
    # # x_train = np.array([np.array(Image.fromarray(te).resize((224, 224), Image.ANTIALIAS)) for te in x_train])
    # # x_test = np.array([np.array(Image.fromarray(te).resize((224, 224), Image.ANTIALIAS)) for te in x_test])
    # # np.save("./cifar10_train_224.npy", x_train)
    # # np.save("./cifar10_train_label.npy", y_train)
    # # np.save("./cifar10_test_224.npy", x_test)
    # # np.save("./cifar10_test_label.npy", y_test)
    # # pdb.set_trace()

    x_train = np.load("./cifar10_train_224.npy")
    y_train = np.load("./cifar10_train_label.npy")
    x_test = np.load("./cifar10_test_224.npy")
    y_test = np.load("./cifar10_test_label.npy")

    x_train_features = []
    for x in tqdm([0,10000,20000,30000,40000]):
        x_train_features.append(model.predict(preprocess_input(x_train[x:x+10000])))
    x_train_features = np.concatenate(x_train_features, axis=0)

    x_test_features = model.predict(preprocess_input(x_test))

    np.save("../../features/cifar10/train_features.npy", x_train_features)
    np.save("../../features/cifar10/test_features.npy", x_test_features)
    np.save("../../features/cifar10/train_labels.npy", y_train)
    np.save("../../features/cifar10/test_labels.npy", y_test)
    np.save("../../features/cifar10/train_images.npy", x_train)
    np.save("../../features/cifar10/test_images.npy", x_test)