#Import modules
import cv2
import os
import imutils
import numpy as np

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
def detect_face_training(img,face_cascade,eye_cascade):
	#convert the test image to gray scale as opencv face detector expects gray images
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#	gray = cv2.equalizeHist(gray)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	gray = clahe.apply(gray)
	#let's detect multiscale images(some images may be closer to camera than others)
	#result is a list of faces
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=1, minSize=(1, 1), flags = cv2.CASCADE_SCALE_IMAGE);

	#if no faces are detected then return None
	if (len(faces) == 0):
		return None, None
	else:
		#return only the face part of the image
		(x, y, w, h) = faces[0]
#		draw_rectangle(img,(x, y, w, h))
#		cv2.imshow("Faces found" ,img)
#		cv2.waitKey(10)

	return gray[y:y+w, x:x+h], faces[0]


#function to detect face using OpenCV
def detect_face_predict(img,face_cascade,eye_cascade):

	#list to hold all detected faces
	detected_faces = []
	#list to hold all rectangle of faces
	detected_faces_rectangle = []
	#list to hold all rectangle of eyes
	detected_eyes_rectangle = []

	#convert the test image to gray scale as opencv face detector expects gray images
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#	gray = cv2.equalizeHist(gray)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	gray = clahe.apply(gray)
	#let's detect multiscale images(some images may be closer to camera than others)
	#result is a list of faces
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=1, minSize=(1, 1), flags = cv2.CASCADE_SCALE_IMAGE);

	#if no faces are detected then return original img
	if (len(faces) == 0):
		return None, None, None
	else:
		for (x, y, w, h) in faces:
			gray_face_area=gray[y:y+w, x:x+h]
			eye = eye_cascade.detectMultiScale( gray_face_area, scaleFactor=1.15, minNeighbors=1, minSize=(1, 1), flags = cv2.CASCADE_SCALE_IMAGE)
			if len(eye) != 0:
				detected_faces.append(gray_face_area)
				detected_faces_rectangle.append((x, y, w, h))
				draw_rectangle(gray,(x, y, w, h))

	#return only the parts with faces in the image
	return detected_faces, detected_faces_rectangle, detected_eyes_rectangle

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
#of faces and another list of labels for each face
def prepare_training_data(data_folder_path,face_resolution,face_cascade,eye_cascade):
 
	#------STEP-1--------
	#get the directories (one directory for each subject) in data folder
	dirs = os.listdir(data_folder_path)
 
	#list to hold all subject faces
	faces = []
	#list to hold labels for all subjects
	labels = []

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
		print('I am training for subject: ',label)

		#build path of directory containing images for current subject subject
		#sample subject_dir_path = "training-data/s1"
		subject_dir_path = data_folder_path + "/" + dir_name

		#get the images names that are inside the given subject directory
		subject_images_names = os.listdir(subject_dir_path)

		#------STEP-3--------
		#go through each image name, read image, 
		#detect face and add face to list of faces
		for image_name in subject_images_names:

			#ignore system files like .DS_Store
			if image_name.startswith("."):
				continue;

			#build image path
			#sample image path = training-data/s1/1.pgm
			image_path = subject_dir_path + "/" + image_name

			#read image
			image = cv2.imread(image_path)
			#detect face
			face, rect = detect_face_training(image,face_cascade,eye_cascade)

			#------STEP-4--------
			#for the purpose of this tutorial
			#we will ignore faces that are not detected
			if face is not None:
				# resize face to use different face_recognizers
				face_resized = cv2.resize(face, (face_resolution,face_resolution))
				#add face to list of faces
				faces.append(face_resized)
				#add label for this face
				labels.append(label)

			image = flip_axis(image,1)
			#detect face
			face, rect = detect_face_training(image,face_cascade,eye_cascade)

			#------STEP-4--------
			#for the purpose of this tutorial
			#we will ignore faces that are not detected
			if face is not None:
				# resize face to use different face_recognizers
				face_resized = cv2.resize(face, (face_resolution,face_resolution))
				#add face to list of faces
				faces.append(face_resized)
				#add label for this face
				labels.append(label)

			image = imutils.rotate(image, 35)
			#detect face
			face, rect = detect_face_training(image,face_cascade,eye_cascade)

			#------STEP-4--------
			#for the purpose of this tutorial
			#we will ignore faces that are not detected
			if face is not None:
				# resize face to use different face_recognizers
				face_resized = cv2.resize(face, (face_resolution,face_resolution))
				#add face to list of faces
				faces.append(face_resized)
				#add label for this face
				labels.append(label)

	return faces, labels

# function to draw a rectangle
def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
#function to draw text on given image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

#this function recognizes the person in image passed and
#draws a rectangle around detected face with name of the subject
def predict(test_img,face_resolution,face_recognizer,subjects,face_cascade,eye_cascade):
	#make a copy of the image as we don't want to change original image
	img = test_img.copy()

	#detect face from the image
	faces, rects, cc = detect_face_predict(img,face_cascade,eye_cascade)
	if faces is not None:
		for i in range(len(faces)):
			# resize face to use different face recognizers
			face_resized = cv2.resize(faces[i], (face_resolution,face_resolution))
			#predict the image using our face recognizer 
			label = face_recognizer.predict(face_resized)
			print(label)
			#get name of respective label returned by face recognizer
			if label[1] < 30000:
				label_text = subjects[label[0]]
				#draw a rectangle around face detected
				draw_rectangle(img, rects[i])
				#draw name of predicted person
				draw_text(img, label_text, rects[i][0], rects[i][1]-5)

	return img
