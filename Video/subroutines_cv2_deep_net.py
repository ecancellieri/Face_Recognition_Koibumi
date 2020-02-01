#Import modules
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import keras.backend as K

import cv2
import os
import imutils
import numpy as np
import matplotlib.pyplot as plt

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	c = list(zip(a, b))
	np.random.shuffle(c)
	a, b = zip(*c)

	return a, b

#function to detect face using OpenCV
def detect_face_predict(image,face_cascade_lbp,face_cascade_haar,eye_cascade):

	#list to hold all detected faces
	detected_faces = []
	#list to hold all rectangle of faces
	detected_faces_rectangle = []

	#convert the test image to gray scale as opencv face detector expects gray images
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#let's detect multiscale images. The result is a list of faces
	faces = face_cascade_lbp.detectMultiScale(gray, scaleFactor=1.025, minNeighbors=20, minSize=(1, 1), flags = cv2.CASCADE_SCALE_IMAGE);
#	faces = face_cascade_haar.detectMultiScale(gray, scaleFactor=1.025, minNeighbors=20, minSize=(1, 1), flags = cv2.CASCADE_SCALE_IMAGE);

	#if no faces are detected then return original img
	if (len(faces) == 0):
		return None, None
	else:
		for (x, y, w, h) in faces:
			gray_face_area=gray[y:y+w, x:x+h]
			image_face_area=image[y:y+w, x:x+h]
			eye = eye_cascade.detectMultiScale( gray_face_area, scaleFactor=1.5, minNeighbors=1, minSize=(1, 1), flags = cv2.CASCADE_SCALE_IMAGE)
			if len(eye) != 0:
				detected_faces.append(image_face_area)
				detected_faces_rectangle.append((x, y, w, h))
	#return only the parts with faces in the image
	return detected_faces, detected_faces_rectangle

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
#of faces and another list of labels for each face
def prepare_training_data(data_folder_path,inputShape):
 
	#------STEP-1--------
	#get the directories (one directory for each subject) in data folder
	dirs = os.listdir(data_folder_path)
 
	#list to hold all subject faces
#	faces = []
	#list to hold labels for all subjects
#	labels = []
	jj = 0
	#let's go through each directory and read images within it
	for dir_name in dirs:
		#our subject directories start with letter 's' so
		#ignore any non-relevant directories if any
		if not dir_name.startswith("s"):
			continue;
 
		#------STEP-2--------
		#extract label number of subject from dir_name
		#format of dir name = slabel
		#, so removing letter 's' from dir_name will give us label
		label = int(dir_name.replace("s", ""))
		print('I am loading images for subject: ',label)

		#build path of directory containing images for current subject subject
		#sample subject_dir_path = "training-data/s1"
		subject_dir_path = data_folder_path + "/" + dir_name

		#get the images names that are inside the given subject directory
		subject_images_names = os.listdir(subject_dir_path)

		#------STEP-3--------
		#go through each image name, read image, 
		#and add image to list of images
		for image_name in subject_images_names:
			jj = jj + 1
			#ignore system files like .DS_Store
			if image_name.startswith("."):
				continue;

			#build image path
			#sample image path = training-data/s1/1.pgm
			image_path = subject_dir_path + "/" + image_name

			#read image
			image = load_img(image_path, target_size=inputShape)
			image = img_to_array(image)
			horizontal_img = flip_axis(image,1)
			rotated_image = imutils.rotate(image, 35)

#			plt.imshow(rotated_image/255.)
#			plt.show()

			image = np.expand_dims(image, axis=0)
			horizontal_img = np.expand_dims(horizontal_img, axis=0)
			rotated_image = np.expand_dims(rotated_image, axis=0)
			image = imagenet_utils.preprocess_input(image)
			horizontal_img = imagenet_utils.preprocess_input(horizontal_img)
			rotated_image = imagenet_utils.preprocess_input(rotated_image)

			if jj == 1:
				#add image to set of images
				faces = image
				faces = np.append(faces,horizontal_img,0)
				faces = np.append(faces,rotated_image,0)
				#add label for this face
				labels = label
				labels = np.append(labels,label)
				labels = np.append(labels,label)
			else:
				#add image to set of images
				faces = np.append(faces,image,0)
				faces = np.append(faces,horizontal_img,0)
				faces = np.append(faces,rotated_image,0)
				#add label for this face
				labels = np.append(labels,label)
				labels = np.append(labels,label)
				labels = np.append(labels,label)

	return faces, labels

# function to draw a rectangle
def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
#function to draw text on given image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

