##------------------------------------------------------##
## This python file provides some useful functions used ##
## both in constrcting database and prediction part.    ##
##------------------------------------------------------##

import cv2

background = None # global variable for the background

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
def seg_threshold(image, threshold=10):
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