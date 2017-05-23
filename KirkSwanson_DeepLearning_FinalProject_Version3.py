""" DEEP LEARNING: ASSIGNMENT 3 PART 2 """
""" Kirk Swanson """

""" Import the numpy package """
import numpy as np

""" Import the math package """
import math

""" Package for control over file systems """
import os

""" Timer """
import time

""" Import random package """
import random

""" Import the matplotlib package for plotting """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

""" Import SciPy """
import scipy 
from scipy.misc import imread

""" Load the images and labels as numpy arrays """
images = []
labels = []

for root, subfolders, files in os.walk('.'):
	for f in files:
		if 'full' in f or '.py' in f:
			continue
		images.append(imread(os.path.join(root, f)))
		labels.append(os.path.split(root)[1])

""" Convert images list to a numpy array """
images = np.asarray(images)

""" Extract the labels """
for i in range(len(labels)):
	one_hot = np.zeros(2)
	if float(labels[i][8:]) == 0.05:
		one_hot[0] = 1
		one_hot[1] = 0
	else: 
		one_hot[0] = 0
		one_hot[1] = 1
	labels[i] = one_hot

""" Convert the labels list to a numpy array """
labels = np.asarray(labels)

""" Randomize the data """
permutation = np.random.permutation(20000)
images = images[permutation]
labels = labels[permutation]

""" Reformulate the data into train, validation, and test sets """
test_X = images[0:4000, :, :]
test_Y = labels[0:4000, :]
validation_X = images[4000:7200, :, :]
validation_Y = labels[4000:7200, :]
train_X = images[7200:, :, :]
train_Y = labels[7200:, :]

""" We will use the InteractiveSession class, which interleaves operations that build and run a computation graph """
import tensorflow as tf 
sess = tf.InteractiveSession()

""" Set the learning rate """
eta = 1e-3
batch_size = 50

""" Set the number of iterations """
iterations = 30

""" Define list to hold the cross-entropy loss """
errors = []

""" Input nodes """
x = tf.placeholder(tf.float32, shape=[None, 62500])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

""" Function to initialize the weights with small noise """
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

""" Function to initialize the biases with small positive value """
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

""" Functions to perform convolution and pooling """
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

""" First convolutional layer weight and bias variables for 6 kernels """
W_conv1 = weight_variable([5, 5, 1, 6])
b_conv1 = bias_variable([6])

""" Reshape the input to a 4D tensor, with second and third dimensions as image dimensions and final as color channel """
x_image = tf.reshape(x, [-1, 250, 250, 1])

""" First layer perform convolution and pooling """
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

""" First densely connected layer with RELU activation """
W_fc1 = weight_variable([123*123*6, 300])
b_fc1 = bias_variable([300])

h_pool1_flat = tf.reshape(h_pool1, [-1, 123*123*6])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

""" Second densely connected layer """
W_fc2 = weight_variable([300, 300])
b_fc2 = bias_variable([300])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

""" Output layer """
W_fc3 = weight_variable([300, 2])
b_fc3 = bias_variable([2])

y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

""" Training """
""" Define the average cross-entropy over all examples in a given batch """
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

""" Define training using the ADAM optimizer and cross-entropy loss """
train_step = tf.train.AdamOptimizer(eta).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

""" Reshape validation_X and test_X arrays """
validation_X = validation_X.reshape(len(validation_X), -1)
test_X = test_X.reshape(len(test_X), -1)

for i in range(iterations):
	batch_index = i*batch_size
	batch_X = train_X[batch_index:(batch_index + batch_size), :, :]
	batch_X = batch_X.reshape(len(batch_X), -1)
	batch_Y = train_Y[batch_index:(batch_index + batch_size), :]
	train_accuracy = accuracy.eval(feed_dict={x: batch_X, y_: batch_Y})
	validation_accuracy = accuracy.eval(feed_dict={x: validation_X, y_: validation_Y})
	validation_loss = cross_entropy.eval(feed_dict={x: validation_X, y_: validation_Y})
	#test = y_conv.eval(feed_dict={x: validation_X, y_: validation_Y})
	#print(test)
	#print(y_conv)
	#errors.append(validation_loss)
	print("step %d, training accuracy %g, validation accuracy %g, validation loss %g"%(i, train_accuracy, validation_accuracy, validation_loss))
	train_step.run(feed_dict={x: batch_X, y_: batch_Y})

#errors = np.asarray(errors)
print("test accuracy %g"%accuracy.eval(feed_dict={x: test_X, y_: test_Y}))
print("test loss %g"%cross_entropy.eval(feed_dict={x: test_X, y_: test_Y}))





#1 change convnet to binary classification task
#2 test convnet
#3 rearrange data so that there are no dependent overlaps




