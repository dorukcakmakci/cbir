from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pdb
import sys
import random
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6'
from keras.datasets import cifar100
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
sys.path.insert(1, "../")
from load_data import oxford_images, oxford_labels, oxford_categories
from sklearn.model_selection import train_test_split
from PIL import Image

num_classes = len(oxford_categories)
epochs = 50


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    temp = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(temp):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(8, (4,4), activation="relu")(input)
    x = Conv2D(8, (4,4), activation="relu")(x)
    x = Conv2D(8, (4,4), activation="relu")(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(32, (4,4), activation="relu")(x)
    x = Conv2D(32, (4,4), activation="relu")(x)
    x = Conv2D(32, (4,4), activation="relu")(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (4,4), activation="relu")(x)
    x = Conv2D(128, (4,4), activation="relu")(x)
    x = Conv2D(256, (4,4), activation="relu")(x)
    x = Flatten()(x)
    return Model(input, x)

def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

# # create train and test 
# x_train, x_test, y_train, y_test = train_test_split(oxford_images, oxford_labels, test_size=0.2, stratify=oxford_labels, random_state=6)

# # resize train dataset images to 224 x 224 x 3
# x_train_resized = []
# for tr in x_train:
#     if len(tr.shape) == 2: # grayscale image
#         temp = np.stack((tr,tr,tr), axis=2)
#         x_train_resized.append(np.array(Image.fromarray(temp).resize((224, 224), Image.ANTIALIAS)))
#     else: # RGB image
#         x_train_resized.append(np.array(Image.fromarray(tr).resize((224, 224), Image.ANTIALIAS)))
# x_train_resized = np.array(x_train_resized)

# # resize test dataset images to 224 x 224 x 3
# x_test_resized = []
# for tr in x_test:
#     if len(tr.shape) == 2: # grayscale image
#         temp = np.stack((tr,tr,tr), axis=2)
#         x_test_resized.append(np.array(Image.fromarray(temp).resize((224, 224), Image.ANTIALIAS)))
#     else: # RGB image
#         x_test_resized.append(np.array(Image.fromarray(tr).resize((224, 224), Image.ANTIALIAS)))
# x_test_resized = np.array(x_test_resized)

# y_train = np.array(y_train)
# y_test = np.array(y_test)
# np.save("./oxford_train_224.npy", x_train_resized)
# np.save("./oxford_train_label.npy", y_train)
# np.save("./oxford_test_224.npy", x_test_resized)
# np.save("./oxford_test_label.npy", y_test)

x_train_resized = np.load("./oxford_train_224.npy")
y_train = np.load("./oxford_train_label.npy")
x_test_resized = np.load("./oxford_test_224.npy")
y_test = np.load("./oxford_test_label.npy")



# min max normalize images
x_train_normalized = x_train_resized.astype('float32') / 255.
x_test_normalized = x_test_resized.astype('float32') / 255.
input_shape = x_train_normalized.shape[1:]



# create training+test positive and negative pairs
class_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
temp = min([len(class_indices[d]) for d in range(num_classes)])
tr_pairs, tr_y = create_pairs(x_train_normalized, class_indices)

class_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs(x_test_normalized, class_indices)

# network definition
cnn = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = cnn(input_a)
processed_b = cnn(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
model.compile(loss=contrastive_loss, optimizer="adam", metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=32,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


model.save('../../models/oxford/siamese_more_complex_50epochs.h5')

pdb.set_trace()
