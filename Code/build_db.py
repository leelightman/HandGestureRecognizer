#------------------------------------------------#
## Build database for thresholded hand gestures ##
#------------------------------------------------#

import numpy as np  ## vesion: 1.18.3
import cv2  ## version: 4.1.2
import imutils  ## version: 0.5.3

import os
import sys

background = None # global variable for the background
RESET = False # global vaiable for recompute the background
cur_frame = None # the frame when press 'r' to reset

## Use this function to compute the initial background
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
def seg_threshold(image, threshold=50):
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

	# represents the max number for each gesture in this database
	MAX_NUM = int(sys.argv[1])
	# 'v': victory, 'o': OK, 'h': horn, 'f': fist, 'p': palm
	# more kinds of gestures can be added
	num_picture = {'v':0, 'o':0, 'h':0, 'f':0, 'p':0}
	# there should be a folder called 'data' before running this program
	relative_path = '../data'
	for directory, subdir, files in os.walk(relative_path):
		for file in files:
			# look through all the files already inside this folder
			# if one gesture is full, we don't want to add more images
			if file.startswith('v'):
				num_picture['v'] += 1
			if file.startswith('o'):
				num_picture['o'] += 1
			if file.startswith('h'):
				num_picture['h'] += 1
			if file.startswith('f'):
				num_picture['f'] += 1
			if file.startswith('p'):
				num_picture['p'] += 1

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
			if RESET == False:
				# draw a green box onto clone frame where the hand in
				cv2.rectangle(clone, (box_left, box_top), (box_right, box_bottom), (0,255,0), 3)
				cv2.putText(clone, 'Please put hand in green box', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

				# the details for the max number and the number for each gesture
				cv2.putText(clone, 'Max number for each image: %d' % (MAX_NUM), (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
				# you can see the information on screen
				# when you want to save images, please keep pressing the key
				cv2.putText(clone, 'Press v to save Victory; Cur Number: %d' % (num_picture['v']), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
				cv2.putText(clone, 'Press o to save OK; Cur Number: %d' % (num_picture['o']), (20,135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
				cv2.putText(clone, 'Press h to save Horn; Cur Number: %d' % (num_picture['h']), (20,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
				cv2.putText(clone, 'Press f to save Fist; Cur Number: %d' % (num_picture['f']), (20,205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
				cv2.putText(clone, 'Press p to save Palm; Cur Number: %d' % (num_picture['p']), (20,240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
				cv2.putText(clone, 'Press r to recalibrate the background.', (20,275), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

				gesture_seg = seg_threshold(gray_ROI)

				# only if there IS a hand!
				if gesture_seg:
					(thresh, seg) = gesture_seg

					# draw a contour for the hand in clone frame
					cv2.drawContours(clone, [seg + (box_left, box_top)], -1, (0, 0, 255))
					# show the thresholded iamge for hand in a separated window
					cv2.imshow("Thresholding", thresh)

					# same trick, avoid modifying
					saved_image = thresh.copy()
			else:
				# Use 30 frames after 'r' pressed to recalibrate the background
				if num_frame - cur_frame < 30:
					get_bg(gray_ROI, 0.5)
					cv2.putText(clone, 'Please wait for recalibrating bg...', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
				else:
					# change the RESET to False
					RESET = False
		
		# show the clone frame in a separated window
		cv2.imshow('Frame', clone)

		# wait the input from user to quit or save images
		key_input = cv2.waitKey(1) & 0xff

		if key_input == ord('q'):
			CONT = False
		# reset the background
		elif key_input == ord('r'):
			background = None
			cur_frame = num_frame
			RESET = True
		# save 'v'
		elif key_input == ord('v'):
			if num_picture['v'] < MAX_NUM:
				index = num_picture['v'] + 1
				path = "../data/" + chr(key_input) + "/" + (chr(key_input)+"_"+str(index)+".jpg")
				cv2.imwrite(path, saved_image)
				num_picture['v'] += 1
		#save 'o'
		elif key_input == ord('o'):
			if num_picture['o'] < MAX_NUM:
				index = num_picture['o'] + 1
				path = "../data/" + chr(key_input) + "/" + (chr(key_input)+"_"+str(index)+".jpg")
				cv2.imwrite(path, saved_image)
				num_picture['o'] += 1
		# save 'h'
		elif key_input == ord('h'):
			if num_picture['h'] < MAX_NUM:
				index = num_picture['h'] + 1
				path = "../data/" + chr(key_input) + "/" + (chr(key_input)+"_"+str(index)+".jpg")
				cv2.imwrite(path, saved_image)
				num_picture['h'] += 1
		# save 'f'
		elif key_input == ord('f'):
			if num_picture['f'] < MAX_NUM:
				index = num_picture['f'] + 1
				path = "../data/" + chr(key_input) + "/" + (chr(key_input)+"_"+str(index)+".jpg")
				cv2.imwrite(path, saved_image)
				num_picture['f'] += 1
		# save 'p'
		elif key_input == ord('p'):
			if num_picture['p'] < MAX_NUM:
				index = num_picture['p'] + 1
				path = "../data/" + chr(key_input) + "/" + (chr(key_input)+"_"+str(index)+".jpg")
				cv2.imwrite(path, saved_image)
				num_picture['p'] += 1
	
# release the wencam and destroy all existing windows
cam.release()
cv2.destroyAllWindows()