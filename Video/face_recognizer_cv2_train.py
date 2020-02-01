import cv2
import os
import subroutines
import numpy as np
from imutils import paths

# parameters
face_resolution = 500  # face resolution for different recognizers
resize_ratio = 1.0     # resize ratio for frames
subjects = ["Masayuki Mori", "Yoshiko Kuga"] # names of subjects (s1,s2)
print('Subjects',subjects)

#load OpenCV face & eye detectors, I am using LBP which is fast
#there is also a more accurate but slow: Haar classifier
face_cascade = cv2.CascadeClassifier('/XXX/opencv/data/lbpcascades/lbpcascade_frontalface_improved.xml')
#face_cascade = cv2.CascadeClassifier('/XXX/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('/XXX/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
#face_cascade = cv2.CascadeClassifier('/XXX/opencv/data/haarcascades/haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('/XXX/opencv/data/haarcascades/haarcascade_eye.xml')


#create our face recognizer:
#LBPH, EigenFace, or Fisher
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer = cv2.face.FisherFaceRecognizer_create()


#let's prepare our training data. Data will be in two lists of same size
#one list will contain all the faces and the other will contain labels for each face
print("Preparing data...")
faces, labels = subroutines.prepare_training_data("training-data",face_resolution,face_cascade,eye_cascade)
faces, labels = subroutines.unison_shuffled_copies(faces,labels)
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))
face_recognizer.save('./trainner.yml')




print("Predicting images...")
# opens the video
video_in = cv2.VideoCapture('./sample3.mp4')
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
		# the fram is re-sized and transformed in grayscale
		frame = cv2.resize(frame, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
		predicted_img = subroutines.predict(frame,face_resolution,face_recognizer,subjects,face_cascade,eye_cascade)
		cv2.imshow("Faces found" ,predicted_img)
		cv2.waitKey(1)

		# writes the frame with the rectangles
		video_out.write(predicted_img)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

# Release everything if job is finished
video_in.release()
video_out.release()
cv2.destroyAllWindows()
