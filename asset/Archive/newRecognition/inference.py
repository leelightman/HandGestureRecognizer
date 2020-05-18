import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from deeplab_test import DeepLabV3Plus
import imutils

background = None
def get_bg(image, W):
	global background
	if background is None:
		background = image.copy().astype("float")
		return
	cv2.accumulateWeighted(image, background, W)

print('TensorFlow', tf.__version__)


H, W = 1280, 1280


model = DeepLabV3Plus(H, W)
model.load_weights('person_weights.h5')


def pad_inputs(image, crop_height=H, crop_width=H, pad_value=0):
    dims = tf.cast(tf.shape(image), dtype=tf.float32)
    h_pad = tf.maximum(crop_height - dims[0], 0)
    w_pad = tf.maximum(crop_width - dims[1], 0)
    padded_image = tf.pad(image, paddings=[[0, h_pad], [0, w_pad], [
                          0, 0]], constant_values=pad_value)
    return padded_image, h_pad, w_pad


def resize_preserve_aspect_ratio(image_tensor, max_side_dim):
    img_h, img_w = image_tensor.shape.as_list()[:2]
    min_dim = tf.maximum(img_h, img_w)
    resize_ratio = max_side_dim / min_dim
    new_h, new_w = resize_ratio * img_h, resize_ratio * img_w
    resized_image_tensor = tf.image.resize(
        image_tensor, size=[new_h, new_w], method='bilinear')
    return resized_image_tensor


def prepare_inputs(image, H=H, W=W, maintain_resolution=False):
    # image = tf.io.read_file(image_path)
    # image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32)
    image.set_shape([None, None, 3])
    shape = image.shape.as_list()[:2]
    if maintain_resolution:
        disp_image = image.numpy().copy()
    image = tf.cast(image, dtype=tf.float32)
    resized = False
    if tf.maximum(shape[0], shape[1]) > H:
        resized = True
        image = resize_preserve_aspect_ratio(image, max_side_dim=H)
    image, h_pad, w_pad = pad_inputs(image)
    if not maintain_resolution:
        disp_image = image.numpy().copy()
    image = image[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    return disp_image, tf.cast(image, dtype=tf.float32), np.int32(h_pad.numpy()), np.int32(w_pad.numpy()), resized


def resize_mask(mask, size):
    mask = tf.image.resize(mask[..., None], size, method='nearest')
    return mask[..., 0]


def pipeline(image_path, alpha=0.7, maintain_resolution=False):
    disp_image, image, h_pad, w_pad, resized = prepare_inputs(
        image_path, maintain_resolution=maintain_resolution)
    mask = model(image[None, ...])[0, ..., 0] > 0.5
    mask = tf.cast(mask, dtype=tf.uint8)
    b_h, b_w = (image.shape[:2] - tf.constant([h_pad, w_pad])).numpy()
    disp_mask = mask[:b_h, :b_w].numpy()
    if resized and maintain_resolution:
        disp_mask = resize_mask(disp_mask, disp_image.shape[:2]).numpy()
    else:
        disp_image = disp_image[:b_h, :b_w]
    overlay = disp_image.copy()
    overlay[disp_mask == 0] = [255, 0, 0]
    overlay[disp_mask == 1] = [0, 0, 255]
    cv2.addWeighted(disp_image, alpha, overlay, 1 - alpha, 0, overlay)
    extracted_pixels = disp_image.copy()
    extracted_pixels[disp_mask == 0] = [207, 207, 207]
    return np.uint8(disp_image), np.uint8(np.concatenate([disp_mask[..., None]] * 3, axis=-1) * 255), np.uint8(overlay), np.uint8(extracted_pixels)


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)

    # count the total frames
    num_frame = 0

    # proportion of ROI interms of the whole frame
    wid_left = 0.65
    wid_right = 0.95
    hei_top = 0.1
    hei_bottom = 0.1 + 0.3 * 1.7

    # variable to judge if we need to continue
    CONT = True

    # represents the max number for each gesture in this database
    # MAX_NUM = 500
    # # 'v': victory, 'o': OK, 'h': horn, 'f': fist, 'p': palm
    # # more kinds of gestures can be added
    # num_picture = {'v': 0, 'o': 0, 'h': 0, 'f': 0, 'p': 0}
    # # there should be a folder called 'data' before running this program
    # relative_path = './data'
    # for directory, subdir, files in os.walk(relative_path):
    #     for file in files:
    #         # look through all the files already inside this folder
    #         # if one gesture is full, we don't want to add more images
    #         if file.startswith('v'):
    #             num_picture['v'] += 1
    #         if file.startswith('o'):
    #             num_picture['o'] += 1
    #         if file.startswith('h'):
    #             num_picture['h'] += 1
    #         if file.startswith('f'):
    #             num_picture['f'] += 1
    #         if file.startswith('p'):
    #             num_picture['p'] += 1

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

        # Region of Interest
        ROI = frame[box_top:box_bottom, box_left:box_right]

        # convert to gray scale and blur it
        gray_ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        gray_ROI = cv2.GaussianBlur(gray_ROI, (15, 15), 0)
        # represent the saved image in each frame
        saved_image = None

        # compute the background during the very first 50 frames
        if num_frame <= 50:
            get_bg(gray_ROI, 0.5)
            cv2.putText(clone, 'Please wait for extracting bg...', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                        2)
        else:
            # draw a green box onto clone frame where the hand in
            cv2.rectangle(clone, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 3)
            cv2.putText(clone, 'Please put hand in green box', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 50, 50),
                        2)

            # the details for the max number and the number for each gesture
            # cv2.putText(clone, 'Max number for each image: %d' % (MAX_NUM), (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            #             (0, 0, 255), 2)
            # # you can see the information on screen
            # # when you want to save images, please keep pressing the key
            # cv2.putText(clone, 'Press v to save Victory; Cur Number: %d' % (num_picture['v']), (20, 100),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # cv2.putText(clone, 'Press o to save OK; Cur Number: %d' % (num_picture['o']), (20, 135),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # cv2.putText(clone, 'Press h to save Horn; Cur Number: %d' % (num_picture['h']), (20, 170),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # cv2.putText(clone, 'Press f to save Fist; Cur Number: %d' % (num_picture['f']), (20, 205),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # cv2.putText(clone, 'Press p to save Palm; Cur Number: %d' % (num_picture['p']), (20, 240),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            gesture_seg = pipeline(ROI)

            # only if there IS a hand!
            if gesture_seg:
                # (thresh, seg) = gesture_seg
                result = np.concatenate(gesture_seg, axis=1)
                # draw a contour for the hand in clone frame
                # cv2.drawContours(clone, [seg + (box_left, box_top)], -1, (0, 0, 255))
                # show the thresholded iamge for hand in a separated window
                cv2.imshow("Thresholding", result)

                # same trick, avoid modifying
                saved_image = result.copy()

        # show the clone frame in a separated window
        cv2.imshow('Frame', clone)

        # wait the input from user to quit or save images
        key_input = cv2.waitKey(1) & 0xff

        if key_input == ord('q'):
            CONT = False
        # save 'v'
        # elif key_input == ord('v'):
        #     if num_picture['v'] < MAX_NUM:
        #         index = num_picture['v'] + 1
        #         path = "data/" + (chr(key_input) + "_" + str(index) + ".jpg")
        #         cv2.imwrite(path, saved_image)
        #         num_picture['v'] += 1
        # # save 'o'
        # elif key_input == ord('o'):
        #     if num_picture['o'] < MAX_NUM:
        #         index = num_picture['o'] + 1
        #         path = "data/" + (chr(key_input) + "_" + str(index) + ".jpg")
        #         cv2.imwrite(path, saved_image)
        #         num_picture['o'] += 1
        # # save 'h'
        # elif key_input == ord('h'):
        #     if num_picture['h'] < MAX_NUM:
        #         index = num_picture['h'] + 1
        #         path = "data/" + (chr(key_input) + "_" + str(index) + ".jpg")
        #         cv2.imwrite(path, saved_image)
        #         num_picture['h'] += 1
        # # save 'f'
        # elif key_input == ord('f'):
        #     if num_picture['f'] < MAX_NUM:
        #         index = num_picture['f'] + 1
        #         path = "data/" + (chr(key_input) + "_" + str(index) + ".jpg")
        #         cv2.imwrite(path, saved_image)
        #         num_picture['f'] += 1
        # # save 'p'
        # elif key_input == ord('p'):
        #     if num_picture['p'] < MAX_NUM:
        #         index = num_picture['p'] + 1
        #         path = "data/" + (chr(key_input) + "_" + str(index) + ".jpg")
        #         cv2.imwrite(path, saved_image)
        #         num_picture['p'] += 1

# release the wencam and destroy all existing windows
cam.release()
cv2.destroyAllWindows()
