from scipy.misc import imread
import numpy as np

class Batch():
	def __init__(self, data, im_size=250, label_size=2):
		self.data = data
		self.index = 0
		self.im_size = im_size
		self.label_size = label_size

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
		images = np.zeros((batch_size, self.im_size**2))
		labels = np.zeros((batch_size, self.label_size))
		for i in range(len(batch)):
			row = batch[i]
			path = row['path']
			label = row['label']
			images[i, :] = imread(path).flatten()
			labels[i, :] = [label['t'], label['rho']]
		return images, labels