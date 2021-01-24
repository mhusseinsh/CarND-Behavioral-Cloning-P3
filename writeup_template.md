# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[modelLoss]: ./output_images/model01_history.png "Model Loss"
[nvidiaNet]: ./output_images/NvidiaNet.png "NVIDIA Net"
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
* [`config.yaml`]() contains all configurations used from training and preprocessing parameters
* [`main.py`]() python main script to initialize variables and classes and run the training
* [`preprocess.py`]() containing data loading and preprocessing
* [`visualization.py`]() containing all visualization methods  
* [`model.py`]() containing the script to create and train the model
* [`drive.py`]() for driving the car in autonomous mode
* [`model.h5`]() containing a trained convolution neural network 
* [`README.md`]() or writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The scripts are reusable and readable. 
The fact that all the included files in this repo can be easily reused and generalized. 
This is simply by using the `config.yaml` file which allows the user to define the data path,
preprocessing and training parameters. Once the user is done with finalizing this file, he can
simply run the training through `python main.py config.yaml`. The `main.py` will initialize the 
preprocessing, model, and visualization classes with the predefined parameters in the configurations
and then will start calling the methods for loading, preprocessing data and then training the network.

The `preprocess.py` file contains all data loading and preprocessing methods, it also has the option to
download the data from any external cloud drive using the `wget` which is defined in the shell script
`getData.sh` if and only if the data is not existing in the directory.

The `model.py` file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The `visualization.py` file contains all visualization functions to save the network architecture,
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

This all can found in [`model.py`]() in the [`create_model()`]() function.

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
* Original dataset provided by udacity
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
simply they implemented this framework for a similar task and they produced really good results, 
so I thought I would take it as a starting point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it contained 
dropout layers and I used more training data.

Then I added a cropping layer in order to ignore the upper part of the image which is the sky
and landscapes, as well as the bottom part which is the vehicle's hood and only
concentrate on the important parts of the images which are the streets.

The final step was to run the simulator to see how well the car was driving around track one. 
There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I 
had to create more generalized training data which I will explain later how I did.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
