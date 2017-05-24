from scipy.misc import imread
import numpy as np

def create_one_hot_mapping(data):
	unique_labels = list(set([tuple(row['label']) for row in data]))
	zeros = np.zeros(len(unique_labels))
	one_hot_mapping = dict()

	for i in range(len(unique_labels)):
		label = unique_labels[i]
		one_hot = np.zeros(len(unique_labels))
		one_hot[i] = 1
		one_hot_mapping[label] = one_hot

	return one_hot_mapping

def convert_to_one_hot(data):
	one_hot_mapping = create_one_hot_mapping(data)
	for row in data:
		row['label'] = one_hot_mapping[tuple(row['label'])]
	return data

class Batch():
	def __init__(self, data, one_hot=False):
		if one_hot:
			self.data = convert_to_one_hot(data)
		else:
			self.data = data

		self.index = 0
		self.im_size = len(imread(data[0]['path']).flatten())
		self.label_size = len(data[0]['label'])

	def next(self, batch_size):
		# get the next batch_size rows from data
		if self.index + batch_size <= len(self.data):
			batch = self.data[self.index:self.index + batch_size]
			self.index += batch_size
		else:
			diff = batch_size - (len(self.data) - self.index)
			batch = self.data[self.index:] + self.data[:diff]
			self.index = diff

		# get the images and labels for the batch
		images = np.zeros((batch_size, self.im_size))
		labels = np.zeros((batch_size, self.label_size))
		for i in range(len(batch)):
			row = batch[i]
			images[i, :] = imread(row['path']).flatten()
			labels[i, :] = row['label']

		return images, labels