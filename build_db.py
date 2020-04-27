import numpy as np
import cv2
import imutils

import os

background = None
def get_bg(image, W):
	global background
	if background is None:
		background = image.copy().astype("float")
		return
	cv2.accumulateWeighted(image, background, W)

def seg_threshold(image, threshold=30):
	global background

	subtraction = cv2.absdiff(background.astype("uint8"), image)

	(_, after_threshold) = cv2.threshold(subtraction, threshold, 255, cv2.THRESH_BINARY)

	(contours, _) = cv2.findContours(after_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) == 0:
		return
	else:
		segmented = max(contours, key=cv2.contourArea)
		return (after_threshold, segmented)

if __name__ == "__main__":
	cam = cv2.VideoCapture(1)

	num_frame = 0

	wid_left = 0.65
	wid_right = 0.95
	hei_top = 0.1
	hei_bottom = 0.1 + 0.3*1.7

	CONT = True

	MAX_NUM = 200
	## 'v': victory, 'o': OK, 'h': horn, 'f': fist, 'p': palm
	num_picture = {'v':0, 'o':0, 'h':0, 'f':0, 'p':0}
	relative_path = './data'
	for directory, subdir, files in os.walk(relative_path):
		for file in files:
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

		frame = imutils.resize(frame, width=1000)
		frame = cv2.flip(frame, 1)

		## Use this clone frame to put some extra objects
		clone = frame.copy()

		height = frame.shape[0]
		width = frame.shape[1]

		box_top = int(height * hei_top)
		box_bottom = int(height * hei_bottom)
		box_left = int(width * wid_left)
		box_right = int(width * wid_right)

		## Region of Interest
		ROI = frame[box_top:box_bottom, box_left:box_right]

		gray_ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
		gray_ROI = cv2.GaussianBlur(gray_ROI, (15,15), 0)

		saved_image = None

		if num_frame <= 50:
			get_bg(gray_ROI, 0.5)
			cv2.putText(clone, 'Please wait for extracting bg...', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
		else:
			cv2.rectangle(clone, (box_left, box_top), (box_right, box_bottom), (0,255,0), 3)
			cv2.putText(clone, 'Please put hand in green box', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

			cv2.putText(clone, 'Max number for each image: %d' % (MAX_NUM), (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
			cv2.putText(clone, 'Press v to save Victory; Cur Number: %d' % (num_picture['v']), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
			cv2.putText(clone, 'Press o to save OK; Cur Number: %d' % (num_picture['o']), (20,135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
			cv2.putText(clone, 'Press h to save Horn; Cur Number: %d' % (num_picture['h']), (20,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
			cv2.putText(clone, 'Press f to save Fist; Cur Number: %d' % (num_picture['f']), (20,205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
			cv2.putText(clone, 'Press p to save Palm; Cur Number: %d' % (num_picture['p']), (20,240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

			gesture_seg = seg_threshold(gray_ROI)

			if gesture_seg:
				(thresh, seg) = gesture_seg

				cv2.drawContours(clone, [seg + (box_left, box_top)], -1, (0, 0, 255))
				cv2.imshow("Thresholding", thresh)

				saved_image = thresh.copy()
		

		cv2.imshow('Frame', clone)

		key_input = cv2.waitKey(1) & 0xff

		if key_input == ord('q'):
			CONT = False
		elif key_input == ord('v'):
			if num_picture['v'] < MAX_NUM:
				index = num_picture['v'] + 1
				path = "data/v/" + (chr(key_input)+"1_"+str(index)+".jpg")
				cv2.imwrite(path, saved_image)
				num_picture['v'] += 1
		elif key_input == ord('o'):
			if num_picture['o'] < MAX_NUM:
				index = num_picture['o'] + 1
				path = "data/o/" + (chr(key_input)+"1_"+str(index)+".jpg")
				cv2.imwrite(path, saved_image)
				num_picture['o'] += 1
		elif key_input == ord('h'):
			if num_picture['h'] < MAX_NUM:
				index = num_picture['h'] + 1
				path = "data/h/" + (chr(key_input)+"1_"+str(index)+".jpg")
				cv2.imwrite(path, saved_image)
				num_picture['h'] += 1
		elif key_input == ord('f'):
			if num_picture['f'] < MAX_NUM:
				index = num_picture['f'] + 1
				path = "data/f/" + (chr(key_input)+"1_"+str(index)+".jpg")
				cv2.imwrite(path, saved_image)
				num_picture['f'] += 1
		elif key_input == ord('p'):
			if num_picture['p'] < MAX_NUM:
				index = num_picture['p'] + 1
				path = "data/p/" + (chr(key_input)+"1_"+str(index)+".jpg")
				cv2.imwrite(path, saved_image)
				num_picture['p'] += 1
	
cam.release()
cv2.destroyAllWindows()