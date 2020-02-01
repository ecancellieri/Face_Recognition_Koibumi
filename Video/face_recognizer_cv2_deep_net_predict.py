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


from imutils import paths
import subroutines_cv2_deep_net
import numpy as np
import cv2
import csv
import os

# parameters
face_resolution = 224  # face resolution for different nets
# Models:
# VGG16, inputShape = (224,224)
# VGG19, inputShape = (224,224)
# ResNet50, inputShape = (224,224)
# InceptionV3, inputShape = (229,229)
# Xception, inputShape = (229,229), TensorFlow ONLY


resize_ratio = 1.0     # resize ratio for frames
subjects = ["Masayuki Mori", "Yoshigo Kuga"] # names of subjects (s1,s2)
labels = np.array(subjects)
num_subjects = len(labels)
print('Subjects',subjects)

#load OpenCV face & eye detectors, I am using LBP which is fast
#there is also a more accurate but slow: Haar classifier
face_cascade_lbp = cv2.CascadeClassifier('/Users/cancellieri/opencv/data/lbpcascades/lbpcascade_frontalface_improved.xml')
face_cascade_haar = cv2.CascadeClassifier('/Users/cancellieri/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('/Users/cancellieri/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
#face_cascade = cv2.CascadeClassifier('/Users/cancellieri/opencv/data/haarcascades/haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('/Users/cancellieri/opencv/data/haarcascades/haarcascade_eye.xml')


# define the network
my_model = Sequential()
my_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
my_model.add(Dense(num_subjects, activation='softmax'))
# load weights from checkpoint
my_model.load_weights("weights.best.hdf5")
# compile the networ
my_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# opens the video
video_in = cv2.VideoCapture('./Koibumi.m4v')
# detects the frame-per-second rates and shape of the video
fps = video_in.get(cv2.CAP_PROP_FPS)             # in float format
length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))  # in float format
width = video_in.get(cv2.CAP_PROP_FRAME_WIDTH)   # in float format
height = video_in.get(cv2.CAP_PROP_FRAME_HEIGHT) # in float format
print('fps and n frames', int(fps), int(length))
print('width and height', int(width), int(height))

# AVI file format
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter('output.avi',fourcc, int(fps), (int(resize_ratio*width),int(resize_ratio*height)))


# while the video is open takes frames and work on them
while(video_in.isOpened()):
	ret, frame = video_in.read()
	if ret == True:
		# Make a copy of the frame to work on
		image_detect_face = frame.copy()
		#detect face from the image
		faces, rects = subroutines.detect_face_predict(image_detect_face,face_cascade_lbp,face_cascade_haar,eye_cascade)
		if faces is not None:
			for i in range(len(faces)):
				face_resized = cv2.resize(faces[i], (face_resolution,face_resolution))
				image_recognize_face = img_to_array(face_resized)
				image_recognize_face = np.expand_dims(image_recognize_face, axis=0)
				image_recognize_face = imagenet_utils.preprocess_input(image_recognize_face)
				# make predictions
				preds = my_model.predict(image_recognize_face)
				preds = preds.tolist()
				preds = preds[0]
				max_value = max(preds)
				if max_value >= 0.35:
					max_index = preds.index(max_value)
					subroutines.draw_rectangle(frame, rects[i])
					subroutines.draw_text(frame, subjects[max_index], rects[i][0], rects[i][1]-5)

		# writes the frame with the rectangles
		video_out.write(frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break


# Release everything if job is finished
video_in.release()
video_out.release()
cv2.destroyAllWindows()












