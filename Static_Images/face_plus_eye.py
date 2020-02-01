import cv2
import numpy as np
from imutils import paths

scaleFactor = 0.5
# path to image and to cascade classifier
imageDir = './Masayuki_Mori'
#cascPath_default = '/XXX/opencv/data/lbpcascades/lbpcascade_frontalface.xml'
#cascPath_forntal = '/XXX/opencv/data/lbpcascades/lbpcascade_frontalface_improved.xml'
#cascPath_profile = '/XXX/opencv/data/lbpcascades/lbpcascade_profileface.xml'
cascPath_default = '/XXX/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
#cascPath_forntal = '/XXX/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml'
#cascPath_profile = '/XXX/opencv/data/haarcascades/haarcascade_profileface.xml'
cascPath_eye = '/XXX/opencv/data/haarcascades/haarcascade_eye.xml'

# creates the Haar cascade classifier by loading the classifier
# data from haarcascade_frontalface_alt.xml
face_cascade_default = cv2.CascadeClassifier(cascPath_default)
#face_cascade_frontal = cv2.CascadeClassifier(cascPath_forntal)
#face_cascade_profile = cv2.CascadeClassifier(cascPath_profile)
eye_cascade = cv2.CascadeClassifier(cascPath_eye)

nfaces=0
nimages=0
for imagePath in paths.list_images(imageDir):
	nimages=nimages+1
	print('n images:',nimages)
# Reads the image and transforms it in gray scale
	image = cv2.imread(imagePath)
	image = cv2.resize(image, None, fx=scaleFactor, fy=scaleFactor, interpolation=cv2.INTER_LINEAR)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray_equ = cv2.equalizeHist(gray)
# Detect faces in the image called "gray" and output
# a set of rectangles for where the faces are
# The detectMultiScale function is a general function that detects objects.
# Since we are calling it on the face cascade, that’s what it detects.
# The first option is the grayscale image. The second is the scaleFactor.
# Since some faces may be closer to the camera, they would appear bigger than
# those faces in the back. The scale factor compensates for this.
# The detection algorithm uses a moving window to detect objects.
# minNeighbors defines how many objects are detected near the current one before
# it declares the face found. minSize, meanwhile, gives the size of each window.
	faces_default = face_cascade_default.detectMultiScale( gray_equ, scaleFactor=1.001, minNeighbors=20, minSize=(1, 1), flags = cv2.CASCADE_SCALE_IMAGE)
#	faces_frontal = face_cascade_frontal.detectMultiScale( gray_equ, scaleFactor=1.22, minNeighbors=3, minSize=(1, 1), flags = cv2.CASCADE_SCALE_IMAGE)
#	faces_profile = face_cascade_profile.detectMultiScale( gray_equ, scaleFactor=1.03, minNeighbors=3, minSize=(1, 1), flags = cv2.CASCADE_SCALE_IMAGE)
#	print('faces',len(faces_default))
	if len(faces_default) != 0:
		for (x, y, w, h) in faces_default:
			gray_face_area=gray_equ[y:y+h, x:x+w]
			eye = eye_cascade.detectMultiScale( gray_face_area, scaleFactor=1.5, minNeighbors=1, minSize=(1, 1), flags = cv2.CASCADE_SCALE_IMAGE)
#			print(len(eye))
			if len(eye) != 0:
				cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
				for (xe, ye, we, he) in eye:
					cv2.rectangle(image, (x+xe, y+ye), (x+xe+we, y+ye+he), (255, 255, 255), 2)
				nfaces = nfaces +1
				print('n faces found:',nfaces)
#				cv2.imshow("Faces found" ,image)
#				cv2.waitKey(0)


# print the number of faces found (i.e. the length of array "faces")
#	print("Found {0} default faces!".format(len(faces_default)))
#	print("Found {0} frontal faces!".format(len(faces_frontal)))
#	print("Found {0} profile faces!".format(len(faces_profile)))
#	print("Found {0} eyes!".format(len(eye)))

# Draw a rectangle around the faces
#	for (x, y, w, h) in faces_default:
#		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
#	for (x, y, w, h) in faces_frontal:
#		cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)
#	for (x, y, w, h) in faces_profile:
#		cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
#	for (x, y, w, h) in eye:
#		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)

#	cv2.imshow("Faces found" ,image)
#	cv2.waitKey(0)
