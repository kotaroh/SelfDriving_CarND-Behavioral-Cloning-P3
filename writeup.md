#**Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (No change from the original one)
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model uses Nvidia architecture, which consists of three convolution neural networks with 5x5 filter sizes and depths between 24 and 48 (model.py lines 55-60) 

The model includes 4 fully-connected layers, and the data is normalized in the model using a Keras lambda layer (model.py line 54). 

####2. Attempts to reduce overfitting in the model

I did not add any code to reduce overfitting. The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 1- 44). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 70).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The sample data added to the project was used as a starting point.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to choose a model from a set of known models which works the best. I started from LeNet, but then realized that Nvidia architecture explained in the video works much better than LeNet in general, after some experiments by using data captured from the simulator.

####2. Final Model Architecture

As described above, Nvidia architecture was adopted.

####3. Creation of the Training Set & Training Process

My original understanding was that  the simulator would be well trained by providing enough data to a correct architecture and failure of the navigation means lack of data. With this assumption, I added more and more data per failure. At the end, I added around 20 laps and 30 data sets, including driving in reverse and recovery from the edge. The number of images used for the training I used was more than 150k. 

It was not possible to process those images on my local machine, so I used a machine with GPU on AWS. At the beginning I used an instance with 8GB memory, then I upgraded to one with 48GB memory.

However, even with the volume of data, my simulator kept failing to navigate one full lap. I thought it should be because of overfitting and then added dropout layers and changed the rate for dropout, but the result did not change.

After three weeks of tries, I switched back to just the sample data from the project without any dropout, which made one lap succeeded. I learned that it is still challenging to recognize how much data is enough and what the best parameter set is.

