#**Behavioral Cloning** 

##Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

[center_lane_driving]: ./writeup_images/center_lane_driving.jpg
[recover_left_1]: ./writeup_images/recover_left_1.jpg
[recover_left_2]: ./writeup_images/recover_left_2.jpg
[recover_left_3]: ./writeup_images/recover_left_3.jpg
[recover_right_1]: ./writeup_images/recover_right_1.jpg
[recover_right_2]: ./writeup_images/recover_right_2.jpg
[recover_right_3]: ./writeup_images/recover_right_3.jpg

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I implemented a modified LeNet architecture that included dropout between the dense layers to prevent overfitting.

My model consists of a convolution neural network with 5x5 filter sizes and depths between 32 and 128 (model.py lines 18-24)  (should I try 32 and 128?)

The model includes RELU layers to introduce nonlinearity (in `lenet` in model.py), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (in `lenet` in model.py). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (in `lenet` in model.py).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the LeNet architecture and increase complexity as needed.
I wanted to avoid making the model more complex than it needed to be so it would train quickly. 
This would allow me to keep a quick feedback cycle and focus on collecting the correct training data.

I chose to use a convolutional neural network because they have proven to be highly effective with image data and we are working with a stream of images.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
To avoid overfitting, I included dropout in the dense layers of the model.

The final step was to run the simulator to see how well the car was driving around track one. 
There are several places where the car would come off the track, mostly in areas where there was an uncommon background feature,
such as a lake, or a dirt path. It seems that the car would either mistake another edge for the edge of the road, or simply did not have
enough experience classifying images with this type of background to properly know what to do. To combat this problem,
I collected more data driving back and forth in this type of environment. This remedied the problem.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (`model.py` in function `lenet` ) 
consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   					    | 
| Convolution 4X4     	| 1x1 stride, valid padding, outputs 156x316x8 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 78x158x8  				|
| Convolution 4x4	    | 1x1 stride, valid padding, outputs 74x154x16  |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 37x77x16  				|
| Flatten               | outputs 45584x1                               |
| Fully connected		| Outputs hidden layer size 120        	        |
| Dropout               | keep_prob = .5 during training                |
| RELU                  |                                               |
| Fully connected		| Outputs hidden layer size 84        			|
| Dropout               | keep_prob = .5 during_training                |
| RELU                  |                                               |
| Fully connected		| Outputs classes size 1        				|
| Dropout               | keep_prob = .5 during_training                |
| Softmax				| Returns probability of each class        		|


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center lane driving][center_lane_driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct if it started going off the road

These images show what a recovery looks like starting from the left (with many frames skipped):

![recover_left_1][recover_left_1]
![recover_left_2][recover_left_2]
![recover_left_3][recover_left_3]

And these images show what a recovery looks like starting from the right (with many frames skipped):

![recover_right_1][recover_right_1]
![recover_right_2][recover_right_2]
![recover_right_3][recover_right_3]

To augment the data sat, I also flipped images and angles thinking that this would would remove any bias for a particular direction of turning.
This occurs in `model.py` in the function `augment_flip`
Because the track is a loop, if you drive around only in the starting direction, your classifier will think it should always turn left. In addition,
I drove around the loop in the other direction.
Its good to flip all of the data to ensure that is balanced, in case I accidentally collected more data in one direction

#### Training

After the collection process, I had 10842 data points. 
I then preprocessed this data by normalizing the data in `model._normalize_pixel`
and I removed unused portions of the image using `Cropping2D`

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
That is, I trained on 8673 samples, validated on 2169 samples.

The validation set helped determine if the model was over or under fitting. 
I trained the model for 10 epochs and only saved the model when it improved on the validation set using `ModelCheckpoint`
This would allow me to train the model as much as it was useful and not have to worry about selecting the perfect number of epochs.
I used an adam optimizer so that manually training the learning rate wasn't necessary.
