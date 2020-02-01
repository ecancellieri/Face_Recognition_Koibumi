#Classify images using pre-trained deep network. Changing the
#last layer and fitting only that one. Code from:
#https://www.kaggle.com/dansbecker/transfer-learning
#
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19

from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import model_from_json
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder


import subroutines_cv2_deep_net
import numpy as np
import csv
import os


np.random.seed(12345)
#len_val_set = 30
size_of_batch = 20
number_of_epochs = 10
dirpath = 'training-data'
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# Models:
# VGG16, inputShape = (224,224)
# VGG19, inputShape = (224,224)
# ResNet50, inputShape = (224,224)
# InceptionV3, inputShape = (229,229)
# Xception, inputShape = (229,229), TensorFlow ONLY



print("Preparing data...")
faces, labels = subroutines.prepare_training_data("training-data",inputShape)
faces, labels = subroutines.unison_shuffled_copies(faces,labels)
# total number of images in training + validation set
ntot = len(labels)
print("Total number of images:", ntot)
len_val_set = np.int(ntot/10)
print("Number of images for validation:", len_val_set)


# convert face labels to dummy variables (i.e. one hot encoded)
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
dummy_y = np_utils.to_categorical(encoded_Y)
num_subjects = np.shape(dummy_y)[1]

# identify train set
dummy_y_val = np.array(dummy_y[0:len_val_set])
X_val = np.array(faces[0:len_val_set])
# identify validation set
dummy_y_train = np.array(dummy_y[len_val_set:ntot])
X_train = np.array(faces[len_val_set:ntot])
print("Data prepared")

print("Preparing network")
# define the network
my_model = Sequential()
my_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
my_model.add(Dense(num_subjects, activation='softmax'))
# Say not to train first layer (ResNet) model. It is already trained
my_model.layers[0].trainable = False

# load weights from checkpoint
# in case of already partially trained
# network with checkpoints
my_model.load_weights("weights.best.hdf5")
# compile the networ
my_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
print("Network prepared")

print("Training the network")
# fit the network
my_model.fit(X_train,dummy_y_train, epochs=number_of_epochs, batch_size=size_of_batch, validation_data=[X_val,dummy_y_val], callbacks=callbacks_list, verbose=2)
#my_model.fit(X_train,dummy_y_train, epochs=number_of_epochs, batch_size=size_of_batch, validation_split=0.1, callbacks=callbacks_list, verbose=2)

