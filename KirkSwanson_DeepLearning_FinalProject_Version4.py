""" DEEP LEARNING: ASSIGNMENT 3 PART 2 """
""" Kirk Swanson """

""" Import the numpy package """
import numpy as np
np.random.seed(0)

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

""" Import json """
import json

""" Import batch class """
from batch import Batch

""" Import tensorflow """
import tensorflow as tf

""" Load the metadata """
data = json.load(open('metadata/metadata.json', 'r'))
np.random.shuffle(data)

""" Restrict data """
data_size = 500
temps = [0.5]
rhos = [0.05, 0.75]

if temps and rhos:
	data = list(filter(lambda x: x['label'][0] in temps and x['label'][1] in rhos, data))
if data_size:
	data = data[:data_size]

""" Divide data into train, validation, and test """
train_data = data[:int(0.8*len(data))]
validation_data = data[int(0.8*len(data)):int(0.9*len(data))]
test_data = data[int(0.9*len(data)):]

""" Create batch generators for train, validation, and test """
one_hot = True
train = Batch(train_data, one_hot=one_hot)
validation = Batch(validation_data, one_hot=one_hot)
test = Batch(test_data, one_hot=one_hot)

n_outputs = train.label_size

""" We will use the InteractiveSession class, which interleaves operations that build and run a computation graph """
sess = tf.InteractiveSession()

""" Set the learning rate """
eta = 1e-3
batch_size = 50

""" Set the number of iterations """
iterations = 11

""" Input nodes """
x = tf.placeholder(tf.float32, shape=[None, 62500])
y_ = tf.placeholder(tf.float32, shape=[None, n_outputs])

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
W_conv1 = weight_variable([10, 10, 1, 6])
b_conv1 = bias_variable([6])

""" Reshape the input to a 4D tensor, with second and third dimensions as image dimensions and final as color channel """
x_image = tf.reshape(x, [-1, 250, 250, 1])

""" First layer perform convolution """
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

""" Second convolutional layer weight and bias variables for 16 kernels """
W_conv2 = weight_variable([5, 5, 6, 16])
b_conv2 = bias_variable([16])

""" Second layer perform convolution """
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

""" First densely connected layer with RELU activation """
W_fc1 = weight_variable([237*237*16, 80])
b_fc1 = bias_variable([80])

h_conv2_flat = tf.reshape(h_conv2, [-1, 237*237*16])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

""" Output layer """
W_fc2 = weight_variable([80, n_outputs])
b_fc2 = bias_variable([n_outputs])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

""" Training """
""" Define the average loss over all examples in a given batch """
if one_hot:
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
else:
	loss = tf.reduce_mean(tf.square(y_ - y_conv))
	t_loss = tf.reduce_mean(tf.square(y_[:, 0] - y_conv[:, 0]))
	rho_loss = tf.reduce_mean(tf.square(y_[:, 1] - y_conv[:, 1]))

""" Define training using the ADAM optimizer """
train_step = tf.train.AdamOptimizer(eta).minimize(loss)
if one_hot:
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

""" Define lists to hold the losses """
train_iterations = []
train_errors = []
validation_iterations = []
validation_errors = []

print
print "Starting training"
for i in range(iterations):
	train_X, train_Y = train.next(batch_size)
	train_loss = loss.eval(feed_dict={x: train_X, y_: train_Y})
	train_iterations.append(i)
	train_errors.append(train_loss)

	if i % 5 == 0:
		validation_X, validation_Y = validation.next()
		validation_loss = loss.eval(feed_dict={x: validation_X, y_: validation_Y})
		validation_iterations.append(i)
		validation_errors.append(validation_loss)

		if one_hot:
			validation_accuracy = accuracy.eval(feed_dict={x: validation_X, y_: validation_Y})
			print("step {}, validation accuracy {:.4f}, validation loss {:.4f}".format(i, validation_accuracy, validation_loss))
		else:
			validation_t_loss = t_loss.eval(feed_dict={x: validation_X, y_: validation_Y})
			validation_rho_loss = rho_loss.eval(feed_dict={x: validation_X, y_: validation_Y})
			print("step {}, validation loss {:.4f}, validation_t_loss {:.4f}, validation_rho_loss {:.4f}".format(i, validation_loss, validation_t_loss, validation_rho_loss))
		print

	if one_hot:
		train_accuracy = accuracy.eval(feed_dict={x: train_X, y_: train_Y})
		print("step {}, training accuracy {:.4f}, training loss {:.4f}".format(i, train_accuracy, train_loss))
	else:
		train_t_loss = t_loss.eval(feed_dict={x: train_X, y_: train_Y})
		train_rho_loss = rho_loss.eval(feed_dict={x: train_X, y_: train_Y})
		print("step {}, training accuracy {:.4f}, training loss {:.4f}, training t_loss {:.4f}, training rho_loss {:.4f}".format(i, train_loss, train_t_loss, train_rho_loss))

	train_step.run(feed_dict={x: train_X, y_: train_Y})

test_X, test_Y = test.next()
if one_hot:
	print("test accuracy {:.4f}".format(accuracy.eval(feed_dict={x: test_X, y_: test_Y})))
print("test loss {:.4f}".format(loss.eval(feed_dict={x: test_X, y_: test_Y})))

""" Plot results """
plt.subplot(121)
plt.plot(train_iterations, train_errors)
plt.title("Train loss")
plt.ylabel('Loss')
plt.xlabel('Number of iterations')

plt.subplot(122)
plt.plot(validation_iterations, validation_errors)
plt.title("Validation loss")
plt.ylabel('Loss')
plt.xlabel('Number of iterations')

plt.show()
plt.savefig('error.png')