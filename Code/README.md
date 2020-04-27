# Generate gesture database
This code will create a gesture database in white-black format in our local machine mainly using ```numpy``` and ```open-cv```. To make this code reproducible, the dependencies of these packages are included inside the code. Also, the code are highly annotated, it's easier to understand the whole logic process.
## What this program actually do:
* This code give you a interaction using webcam of your computer. Once your camera is open, you can put your hand into a specified region and then press some keys to save the image to a local folder.
* There are five kinds of gestures you can save and users can totally add more gestures into the code as they want.
* The format of saved image is not RGB or the colored image, instead, the white-black image will be saved. This will undoubtedly benefit the training process later. To do this, we use lots of built-in functions from open-cv. Users can find more details when looking through the code.
## How to run this code:
* First, under the same path you put the code, please create a folder named 'data' where the images are stored.
* Then, please run: ```python build_db.py 200```. The second parameter means the maximum number of each geasture you want. In this case, I want 200 images for each of these five gestures.
* Finally, you will see the saved images inside data folder. All of these images are well named. For example, the ```fist``` gesture will start with ```f_num``` where num range from ```[1,200]```.
## Some samples:

![image](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/samples/o_1.jpg)
![image](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/samples/o_12.jpg)

![image](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/samples/p_3.jpg)
![image](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/samples/p_4.jpg)

![image](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/samples/f_2.jpg)
![image](https://github.com/leelightman/HandGestureRecognizer/blob/master/Code/samples/f_9.jpg)
