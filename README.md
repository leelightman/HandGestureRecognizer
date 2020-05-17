# Hand Gesture Recognition

## Team:

Dongzi Qu (dq394), Lynn Li (ml6589), Mina Lee (ml6543), Zili Xie (zx979)

## Description:

- Recognize multiple hand gestures from live videos using webcam of users' computers.
- Interpret the meaning of the hand gestures. For example, ‘V’, thumbs up, ‘OK’, the horn fingers, the fist bump, and the high five.

## Methods:

- Segmentation
- Background subtraction
- Active contour
- Thresholding
- Deep learning

## Library (mainly):

- OpenCV
- Keras
- NumPy

## How to use:

- First, go the `models` folder to download the models from our drive following the instructions, and save them to `models`.
- `cd Code`
- `python predict.py <model_name>` (_model_name_ refers to the model you want to use)

## Some Results:

Here are some testing samples, and for more results, you can find from inside `Code/samples` or watch this great [video](https://drive.google.com/file/d/1uDyKOb6X-R4Ju5196eYVg2bNBCSeRFZI/view)
![image](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/samples/readme_o.png)
![image](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/samples/readme_r.png)

---

# Outline

## Data: [build_db.py](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/build_db.py) (Generate gesture database)

This code will create a gesture database in white-black format in our local machine mainly using `numpy` and `open-cv`. To make this code reproducible, the dependencies of these packages are included inside the code. Also, the code are highly annotated, it's easier to understand the whole logic process.

#### What this program actually do:

- This code give you a interaction using webcam of your computer. Once your camera is open, you can put your hand into a specified region and then press some keys to save the image to a local folder.
- There are five kinds of gestures you can save and users can totally add more gestures into the code as they want.
- The format of saved image is not RGB or the colored image, instead, the white-black image will be saved. This will undoubtedly benefit the training process later. To do this, we use lots of built-in functions from open-cv. Users can find more details when looking through the code.
- Last but not the least, if you move your computer or your background changed, you can press 'r' to recalibrate your background inside the ROI (Region of Interest). In that case, users don't need to re-complie the whole program to build their database. Also, this tool can be used when the background is not so good for extracting ideal gestures.

#### How to run this code:

- First, under the same path you put the code, please create a folder named 'data' where the images are stored.
- Then, please run: `python build_db.py 200`. The second parameter means the maximum number of each geasture you want. In this case, I want 200 images for each of these five gestures.
- Finally, you will see the saved images inside data folder. All of these images are well named. For example, the `fist` gesture will start with `f_num` where num range from `[1,200]`.

#### Some samples from database:

![image](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/samples/o_1.jpg)
![image](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/samples/o_12.jpg)

![image](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/samples/p_3.jpg)
![image](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/samples/p_4.jpg)

![image](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/samples/f_2.jpg)
![image](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/samples/f_9.jpg)

## Model 1: [VGG16_additional_layers.ipynb](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/VGG16_additional_layers.ipynb) (recognition model)

This model is built mainly based on the pre-trained model `VGG16` from `keras`. Since the shape of output of the original model (VGG16) is (1000, 1), which can used to classify 1000 labels. However, there are only five different kinds of gestures we have, so we need to transform this 1000 dimension to 5.

#### What this program actually do:

- Adpated from VGG16 and add more linear layers after it to transform the output to (5, 1)
- The code inside notebook is highly annotated, so it's not so hard to interpret.

## Model 2: [handGestureVGGModel.ipynb](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/handGestureVGGModel.ipynb) (recognition model)

For model 2, we try not to use the pre-trained model. Instead, we plan to build our model follow the same ways as VGG16 but change the order of layers and resize the output shape between several layers. The detailed framework of this model can be found inside `VGG16.md`.

## Prediction: [predict.py](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/predict.py)

This code will generate the prediction for the new input gestures based on the model trained inside the notebook, which can be found inside the 'models' folder. Notice that there is more than one model inside this folder, so users can change whatever they want to make prediction. As for now, out models support 5 different kinds of gestures for predicting, including 'palm', 'OK', 'V', 'Rock' and 'fist'. However, it's kind of simple to increase this number. Users just need to do a little midification just inside the build_db.py and notebook.

#### What this program actually do:

- To avoid the frame latency, we predict the gesture once per 5 frames. So, once users put their hands inside the ROI, they might need to wait half a second to be presented the name of gestures on the screen.
- Similar to what we did in `build_db.py`, we also implment the functionality of recalibration in case users move their computer by accident. Accordingly, you don't have to re-run your program, which can be time-consuming because loading the trained model.

#### Some predicting examples:

Results are shown at the beginning.
