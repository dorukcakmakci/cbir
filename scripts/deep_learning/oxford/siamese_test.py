import pdb
import numpy as np
import sys
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6'
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model, load_model
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import backend as K
from tqdm import tqdm
from load_data import oxford_images, oxford_labels, oxford_categories

# workaround for custom loss function
import keras.losses
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
keras.losses.contrastive_loss = contrastive_loss

x_train_resized = np.load("./oxford_train_224.npy")
y_train = np.load("./oxford_train_label.npy")
x_test_resized = np.load("./oxford_test_224.npy")
y_test = np.load("./oxford_test_label.npy")

# min max normalize images
x_train = x_train_resized.astype('float32') / 255.
x_test = x_test_resized.astype('float32') / 255.


# load model
siam = load_model("../../models/oxford/siamese_50epochs.h5")
model = Model(siam.input, siam.output)

# #------------------ computation for each test image -------------------------------------
# # create pairs matrix 
# pairs = []
# for image, label in tqdm(zip(x_test, y_test)):
#     test = np.repeat(image[np.newaxis,:,:,:], x_train.shape[0], axis=0)
#     pairs += [[tr,te] for tr,te in zip(x_train,test)]
# pairs = np.array(pairs)
# pdb.set_trace()
# # perform query for test images batch by batch
# preds = []
# step_size = 50
# prev_step = 0
# for cur_step in tqdm(range(0, 10000*50000 + 1, 50000*step_size)):
#     preds += [np.reshape(model.predict([pairs[prev_step:cur_step,0], pairs[prev_step:cur_step,1]]), (25,-1))]
#     prev_step = cur_step
# pdb.set_trace()


#-------------------- computation for random sampled 10 images per class
def query(image, label):
    # predict
    test = np.repeat(image[np.newaxis,:,:,:], x_train.shape[0], axis=0)
    pairs = np.array([[tr,te] for tr,te in zip(x_train,test)])
    preds = np.reshape(model.predict([pairs[:,0], pairs[:,1]]), (-1,))
    # calculate AP@1, AP@3 and AP@10
    top10 = preds.argsort()[0:10]
    ap1 = 0
    ap3 = 0
    ap10 = 0
    for iter, idx in enumerate(top10):
        if y_train[idx] == label:
            if iter < 1:
                ap1 += 1
            if iter < 3:
                ap3 += 1
            ap10 += 1
    return ap1, ap3 / 3, ap10 / 10

# # randomly sample test images (10 image per class)
# num_classes = len(oxford_categories)
# class_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
# randomly_sampled_images = []
# for arr in class_indices:
#     randomly_sampled_images.append(np.random.choice(arr, size=10))
# samples = np.concatenate(randomly_sampled_images)
# np.save("./randomly_sampled_test_image_idx_oxford.npy", samples)

# # load randomly sampled data 
# samples = np.load("./randomly_sampled_test_image_idx_oxford.npy")

x_sample = x_test[samples]
y_sample = y_test[samples]

# test the content based image retrieval model
ap_1, ap_3, ap_10 = [], [], []
for image, label in tqdm(zip(x_sample, y_sample)):
    ap1, ap3, ap10 = query(image, label)
    ap_1.append(ap1)
    ap_3.append(ap3)
    ap_10.append(ap10)
pdb.set_trace()
print(f"mAP@1 = {np.mean(ap_1)}\nmAP@3 = {np.mean(ap_3)}\nmAP@10 = {np.mean(ap_10)}")





