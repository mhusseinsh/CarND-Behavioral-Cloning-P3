# **Behavioral Cloning** 

## Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[modelViz]: ./output_images/model_plot.png "Model Visualization"
[modelViz1]: ./output_images/output.png "Model Visualization1"
[modelLoss]: ./output_images/model_history.png "Model Loss"
[nvidiaNet]: ./output_images/NvidiaNet.png "NVIDIA Net"
[dataset]: ./output_images/carnd-using-multiple-cameras.png "Udacity Dataset"

[sddCenter]: ./output_images/center_2016_12_01_13_39_25_398.jpg "Center"
[sddLeft]: ./output_images/left_2016_12_01_13_39_25_398.jpg "Left"
[sddRight]: ./output_images/right_2016_12_01_13_39_25_398.jpg "Right"

[track2Center]: ./output_images/center_2021_01_24_14_01_54_258.jpg "Center"
[track2Left]: ./output_images/left_2021_01_24_14_01_54_258.jpg "Left"
[track2Right]: ./output_images/right_2021_01_24_14_01_54_258.jpg "Right"

[2lapsCenter]: ./output_images/center_2021_01_23_21_00_13_761.jpg "Center"
[2lapsLeft]: ./output_images/left_2021_01_23_21_00_13_761.jpg "Left"
[2lapsRight]: ./output_images/right_2021_01_23_21_00_13_761.jpg "Right"

[track2Center_flipped]: ./output_images/center_2021_01_24_14_01_54_258_flipped.jpg "Center"
[track2Left_flipped]: ./output_images/left_2021_01_24_14_01_54_258_flipped.jpg "Left"
[track2Right_flipped]: ./output_images/right_2021_01_24_14_01_54_258_flipped.jpg "Right"

[2lapsCenter_flipped]: ./output_images/center_2021_01_23_21_00_13_761_flipped.jpg "Center"
[2lapsLeft_flipped]: ./output_images/left_2021_01_23_21_00_13_761_flipped.jpg "Left"
[2lapsRight_flipped]: ./output_images/right_2021_01_23_21_00_13_761_flipped.jpg "Right"

[lap1RecoveryCenter]: ./output_images/lap1recoveryCenter.gif "Center"
[lap1RecoveryLeft]: ./output_images/lap1recoveryLeft.gif "Left"
[lap1RecoveryRight]: ./output_images/lap1recoveryRight.gif "Right"

[lap2RecoveryCenter]: ./output_images/lap2recoveryCenter.gif "Center"
[lap2RecoveryLeft]: ./output_images/lap2recoveryLeft.gif "Left"
[lap2RecoveryRight]: ./output_images/lap2recoveryRight.gif "Right"

[1lapccwCenter]: ./output_images/center_2021_01_23_21_05_36_624.jpg "Center"
[1lapccwLeft]: ./output_images/left_2021_01_23_21_05_36_624.jpg "Left"
[1lapccwRight]: ./output_images/right_2021_01_23_21_05_36_624.jpg "Right"

[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [`config.yaml`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/config.yaml) contains all configurations used from training and preprocessing parameters
* [`main.py`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/main.py) python main script to initialize variables and classes and run the training
* [`preprocess.py`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/preprocess.py) containing data loading and preprocessing
* [`visualization.py`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/visualization.py) containing all visualization methods  
* [`model.py`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [`drive.py`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* [`model.h5`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/model.h5) containing a trained convolution neural network 
* [`README.md`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/README.md) or writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The scripts are reusable and readable. 
The fact that all the included files in this repo can be easily reused and generalized. 
This is simply by using the [`config.yaml`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/config.yaml) file which allows the user to define the data path,
preprocessing and training parameters. Once the user is done with finalizing this file, he can
simply run the training through `python main.py config.yaml`. The [`main.py`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/main.py) will initialize the 
preprocessing, model, and visualization classes with the predefined parameters in the configurations
and then will start calling the methods for loading, preprocessing data and then training the network.

The [`preprocess.py`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/preprocess.py) file contains all data loading and preprocessing methods, it also has the option to
download the data from any external cloud drive using the `wget` which is defined in the shell script
[`getData.sh`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/getData.sh) if and only if the data is not existing in the directory.

The [`model.py`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/model.py) file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The [`visualization.py`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/visualization.py) file contains all visualization functions to save the network architecture,
and loss plots.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model architecture was based on the [NVIDIA's DAVE-2 architecture](https://arxiv.org/pdf/1604.07316v1.pdf).

The NVIDIA's model was very successful in performing End to End Learning for Self-Driving Cars by predicting the steering angle.
That's why I chose this model as my base model due to its fantastic results.

![alttext][nvidiaNet]

The first layer is a Keras Lambda layer for input normalisation. 
The input images are normalized to have zero mean and equal variance.
```python
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=self.img_shape))
```
Then it is followed with a Keras Cropping2D layer which crops the top and the bottom of the images
to disregard the unneeded pixels.
```python
model.add(Cropping2D(cropping=((self.top_crop, self.bottom_crop), (0, 0))))
```
It is then followed by 5 Convolution2D layers, then a Flatten layer and three Dense layers with 
RELU activations to introduce nonlinearity and then dropout layers.

This all can found in [`model.py`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/master/model.py) in the [`create_model()`](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/2e048be9766d6f1fc4c1034775cd55621a62b7a8/model.py#L29) function.

#### 2. Attempts to reduce overfitting in the model

As mentioned above, 3 dropout layers were added after each fully connected layer to ensure that 
no overfitting will occur.
```python
model.add(Dropout(0.3))
```
The model was trained and validated on different data sets to ensure that the model was not overfitting.

The model was tested by running it through the simulator on the 2 tracks 
and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model a mean squared error loss and an adam optimizer, so the learning rate was not tuned manually.
```python
model.compile(loss=self.loss, optimizer=self.optimizer)
```

#### 4. Appropriate training data

Training data was chosen wisely to ensure that the vehicle is driving on the road. 
I used a combination of different datasets. This includes:
* Sample driving data provided by udacity
* 2 laps of centre lane driving
* 1 lap of smooth curve driving
* 1 lap of driving counter clockwise
* 1 recovery lap
* 2 laps of centre lane driving from the second track
* 1 recovery lap from the second track 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with simple baby steps.
This involved only implementation of a fully connected layer which takes the image as an input
and outputs only 1 value which is the steering angle. With this simple architecture,
I was able to develop a complete working end-to-end framework which involves reading
training data, performing preprocessing, training a network, extracting the output results,
plotting the losses and then testing my trained model against the simulation and checking the
output.

After this pipeline became ready, I started to replace the architecture with a convolutional
neural network which was provided by NVIDIA as I mentioned above. I thought this model might be appropriate because
simply they implemented this framework for a similar task, and they produced extremely good results, 
so I thought I would take it as a starting point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it contained 
dropout layers, and I used more training data.

Then I added a cropping layer in order to ignore the upper part of the image which is the sky
and landscapes, as well as the bottom part which is the vehicle's hood and only
concentrate on the important parts of the images which are the streets.

The final step was to run the simulator to see how well the car was driving around track one. 
There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I 
had to create more generalized training data which I will explain later how I did.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture as shown [here](https://github.com/mhusseinsh/CarND-Behavioral-Cloning-P3/blob/2e048be9766d6f1fc4c1034775cd55621a62b7a8/model.py#L29)
consisted of a convolution neural network with the following layers and layer sizes:

| Layer (type)        | Output Shape   |   Param #   |
|:-------------------:|:--------------:|:--------------:| 
| lambda_1 (Lambda)      | (None, 160, 320, 3)        |  0        | 
| cropping2d_1 (Cropping2D)     | (None, 90, 320, 3)      | 0        | 
| conv2d_1 (Conv2D)     | (None, 43, 158, 24)      | 1824        | 
| conv2d_2 (Conv2D)     | (None, 20, 77, 36)      | 21636        | 
| conv2d_3 (Conv2D)     | (None, 8, 37, 48)      | 43248        | 
| conv2d_4 (Conv2D)     | (None, 6, 35, 64)      | 27712        | 
| conv2d_5 (Conv2D)     | (None, 4, 33, 64)      | 36928        | 
| flatten_1 (Flatten)    | (None, 8448)      | 0        | 
| dense_1 (Dense)    | (None, 100)      | 844900        | 
| activation_1 (Activation)    | (None, 100)       | 0        | 
| dropout_1 (Dropout)    | (None, 100)      | 0        | 
| dense_2 (Dense)    | (None, 50)      | 5050        | 
| activation_2 (Activation)    | (None, 50)       | 0        | 
| dropout_2 (Dropout)    | (None, 50)      | 0        | 
| dense_3 (Dense)    | (None, 10)      | 510        | 
| activation_3 (Activation)    | (None, 10)       | 0        | 
| dropout_3 (Dropout)    | (None, 10)      | 0        | 
| dense_4 (Dense)   | (None, 1)      | 11        | 

This makes a:
* Total params: 981,819
* Trainable params: 981,819
* Non-trainable params: 0

Here is a visualization of the architecture:

![alt text][modelViz1]


And a more detailed visualization showing the layers' sizes:

![alt text][modelViz]

#### 3. Creation of the Training Set & Training Process

The dataset which can be logged directly from the simulator contain three cameras images (left, center, right), and output labels which in this case will be the steering angles.

![alt text][dataset]

In the beginning, I started with the [sample driving data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided by Udacity's team
To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

|                | Left camera                             | Center camera                               | Right camera                              |
| -------------- | --------------------------------------- | ------------------------------------------- | ----------------------------------------- |
| Original image | ![alt text][sddLeft] | ![alt text][sddCenter] | ![alt text][sddRight] |

I started training my network on only center images, but I figured out that the results are not satisfying and the vehicle is getting off-road.

The simulator captures images from three cameras mounted on the car: a center, right and left camera. That’s because of the issue of recovering from being off-center.

So the strategy is changed, and I started to use the three images.
During training, I fed the left and right camera images to the model as if they were coming from the center camera.
This way, I can teach the model how to steer if the car drifts off to the left or the right.

However, in order to use the two other images, a small steering correction has to be made to the steering angle.

I used a steering correction of (+0.2, 0, -0.2).

I then logged two complete full laps of the first track while only trying to drive in the center.

|                | Left camera                             | Center camera                               | Right camera                              |
| -------------- | --------------------------------------- | ------------------------------------------- | ----------------------------------------- |
| Original image | ![alt text][2lapsLeft] | ![alt text][2lapsCenter] | ![alt text][2lapsRight] |


Moreover, I did a one lap logging of driving on a smoothly around curves.

Afterwards the vehicle was performing better, but during the cases it gets off-road, sometimes it fails to drive back again to the center
I then recorded the vehicle recovering from the left side and right sides of the road back
to center so that the vehicle would learn to what it should do when it gets off-road.
These images show what a recovery looks like:

|                | Left camera                             | Center camera                               | Right camera                              |
| -------------- | --------------------------------------- | ------------------------------------------- | ----------------------------------------- |
| Original image | ![alt text][lap1RecoveryLeft] | ![alt text][lap1RecoveryCenter] | ![alt text][lap1RecoveryRight] |

In order to generalize the model more, I logged a driving scene where I am driving counter clock-wise.
This would be helping the model to train better, as most of the curves were turning left, which
resulted in producing a negative steering angle. So in order to train the vehicle to do a positive
steering in case of curves turning right, I had to log a scenario of driving the other way around.

|                | Left camera                             | Center camera                               | Right camera                              |
| -------------- | --------------------------------------- | ------------------------------------------- | ----------------------------------------- |
| Original image | ![alt text][1lapccwLeft] | ![alt text][1lapccwCenter] | ![alt text][1lapccwRight] |

The model was perfectly working on the first track after the training operation.
The vehicle was able to drive the whole track while remaining on the center of the lane.

![alt text][track1model0]

However, when testing on the second track, the model failed a little bit to drive smoothly around the center.
It was getting off-road so often and was swerving a lot. This is probably the model overfitted itself to the first track.
It learnt it quite good, and failed to generalize on the second track due to the difference of the road shape,
 as well as the landscape and so on.

![alt text][track2model0]

In order to overcome this problem, I repeated almost the same data logging but for second track.

|                | Left camera                             | Center camera                               | Right camera                              |
| -------------- | --------------------------------------- | ------------------------------------------- | ----------------------------------------- |
| Original image | ![alt text][track2Left] | ![alt text][track2Center] | ![alt text][track2Right] |

Also I logged a recovery scenario like I did in the first track

|                | Left camera                             | Center camera                               | Right camera                              |
| -------------- | --------------------------------------- | ------------------------------------------- | ----------------------------------------- |
| Original image | ![alt text][lap2RecoveryLeft] | ![alt text][lap2RecoveryCenter] | ![alt text][lap2RecoveryRight] |

To augment the dataset, I also flipped images and angles thinking that this would 
be an effective technique for helping with the left turn bias.

The process involved flipping images and taking the opposite sign of the steering measurement.
For example, here is an image that has then been flipped:

Track 1:

|                | Left camera                             | Center camera                               | Right camera                              |
| -------------- | --------------------------------------- | ------------------------------------------- | ----------------------------------------- |
| Original image | ![alt text][2lapsLeft] | ![alt text][2lapsCenter] | ![alt text][2lapsRight] |
| Flipped image | ![alt text][2lapsLeft_flipped] | ![alt text][2lapsCenter_flipped] | ![alt text][2lapsRight_flipped] |


Track 2:

|                | Left camera                             | Center camera                               | Right camera                              |
| -------------- | --------------------------------------- | ------------------------------------------- | ----------------------------------------- |
| Original image | ![alt text][track2Left] | ![alt text][track2Center] | ![alt text][track2Right] |
| Flipped image | ![alt text][track2Left_flipped] | ![alt text][track2Center_flipped] | ![alt text][track2Right_flipped] |


After the collection process, I had 57426 number of data points.
I then preprocessed this data by performing flipping operations, so it produced double size
of this number as the training data.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. 

The validation set helped determine if the model was over or under fitting. 

The ideal number of epochs was 10 as evidenced by that the validation loss was not increasing afterwards,
and it actually started to oscillate around a certain value between the 8th and the
12th epochs.

I used an adam optimizer so that manually training the learning rate wasn't necessary.

A plot of the training and validation losses is shown below:

![alt text][modelLoss]

Finally the model outperformed on both tracks with always keeping in the center as shown below:

Track 1:

![alt text][track1model]

Track 2:

![alt text][track2model]


