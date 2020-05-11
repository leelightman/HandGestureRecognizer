#-------------------------------------------------------------------------------#
## Use this code to predict the incoming gestures via a trained model          ##
## The model is constructed and trained mainly based on VGG16 with more layers ##
#-------------------------------------------------------------------------------#

import numpy as np  ## vesion: 1.18.3
import cv2  ## version: 4.1.2
import imutils  ## version: 0.5.3

import os
import sys

# keras version: 2.3.1
from keras.models import load_model
from keras.preprocessing.image import image
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# Use a var outside the excution in order to keep the name between two predictions
gesture_name = None

# Use this function to predict
def predict_gesture(input_image, model):
	x = input_image.copy() # Not change the original image

	# Resize the image to fit our pre-trained model
	# This part are relied on the model itself
	x = cv2.resize(x, (224,224)) 
	x = np.stack((x,)*3, axis=-1)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	result = model.predict(x)
	result = result.flatten() # shape from (5,1) to (5,)
	return np.argmax(result) # output the index with the largest value

# load the trained model
model = load_model('models/saved_model.hdf5')

# mapping between the index and gesture name
gesture_map = ['Fist', 'Rock', 'OK', 'Palm', 'Victory']

## global variable for the background
background = None
def get_bg(image, W):
	global background
	# for the very beginning case
	if background is None:
		background = image.copy().astype("float") # use copy() to avoid changing the orginal image
		return
	# use this function to combine the new image and the 
	# existing background together in some weight
	cv2.accumulateWeighted(image, background, W)

## use this function to get the thresholded image and segmented image for hand
def seg_threshold(image, threshold=5):
	global background

	# substraction between background and the hand image
	subtraction = cv2.absdiff(background.astype("uint8"), image)

	# for the pixel in substraction, larger then threshold will be assigned to 255, otherwise 1
	(_, after_threshold) = cv2.threshold(subtraction, threshold, 255, cv2.THRESH_BINARY)

	# draw the contours, copy() to avoid modifying
	(contours, _) = cv2.findContours(after_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) == 0:
		return
	else:
		# we only need the largest contour
		segmented = max(contours, key=cv2.contourArea)
		return (after_threshold, segmented)

if __name__ == "__main__":
	cam = cv2.VideoCapture(0)

	# count the total frames
	num_frame = 0

	# proportion of ROI interms of the whole frame
	wid_left = 0.65
	wid_right = 0.95
	hei_top = 0.1
	hei_bottom = 0.1 + 0.3*1.7

	# variable to judge if we need to continue
	CONT = True

	while CONT:
		(ret, frame) = cam.read()
		num_frame += 1

		# resize the frame
		frame = imutils.resize(frame, width=1000)
		# flip the frame to avoid the mirror view
		frame = cv2.flip(frame, 1)

		## Use this clone frame to put some extra objects
		## VERY IMPORTANT!!!
		clone = frame.copy()

		height = frame.shape[0]
		width = frame.shape[1]

		# the four corners of the region where we put hand in
		box_top = int(height * hei_top)
		box_bottom = int(height * hei_bottom)
		box_left = int(width * wid_left)
		box_right = int(width * wid_right)

		## Region of Interest
		ROI = frame[box_top:box_bottom, box_left:box_right]

		# convert to gray scale and blur it
		gray_ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
		gray_ROI = cv2.GaussianBlur(gray_ROI, (15,15), 0)

		# represent the saved image in each frame
		saved_image = None

		# compute the background during the very first 50 frames
		if num_frame <= 50:
			get_bg(gray_ROI, 0.5)
			cv2.putText(clone, 'Please wait for extracting bg...', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
		else:
			# draw a green box onto clone frame where the hand in
			cv2.rectangle(clone, (box_left, box_top), (box_right, box_bottom), (0,255,0), 3)
			cv2.putText(clone, 'Please put hand in green box', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

			# Extract the hand region from the ROI
			gesture_seg = seg_threshold(gray_ROI)

			# only if there is a hand!!!
			if gesture_seg:
				(thresh, seg) = gesture_seg

				## New added
				# We predict the gesture each 10 frames to avoid the latency
				if (num_frame - 50) % 10 == 0:
					index = predict_gesture(thresh, model)
					gesture_name = gesture_map[index]

				cv2.putText(clone, 'This gesture is: %s' % (gesture_name), (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

				# draw a contour for the hand in clone frame
				cv2.drawContours(clone, [seg + (box_left, box_top)], -1, (0, 0, 255))
				# show the thresholded iamge for hand in a separated window
				cv2.imshow("Thresholding", thresh)

		
		# show the clone frame in a separated window
		cv2.imshow('Frame', clone)

		# wait the input from user to quit or save images
		key_input = cv2.waitKey(100) & 0xff

		if key_input == ord('q'):
			CONT = False
	
# release the wencam and destroy all existing windows
cam.release()
cv2.destroyAllWindows()