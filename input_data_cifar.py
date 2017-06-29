from __future__ import print_function
import gzip
import os
import urllib
import numpy as np
import csv


def loadData(file):
	import cPickle
	fo = open(file,'rb')
	info = cPickle.load(fo)
	fo.close()
	return info['data'], info['labels']

def loadData2(file):
	fo = csv.reader(open(file,'r'))
	data = []
	for l in fo:
		data.append(l)
	data = np.array(data).astype('float32')		
	return data

def conv2Gray(dat):
	img = []
	for i in range(1024):
		img.append(dat[i]*0.2125 + dat[i+1024]*0.7154 + dat[i+1024+1024]*0.0721)
	return np.array(img)



class DataSet(object):
	def __init__(self, images, labels, fake_data=False):
		if fake_data:
  			self._num_examples = 10000
		else:
			#assert images.shape[0] == labels.shape[0], ("images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
			self._num_examples = images.shape[0]		
			# Convert shape from [num examples, rows, columns, depth]
			# to [num examples, rows*columns] (assuming depth == 1)
			# assert images.shape[3] == 1
 			# images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
			# Convert from [0, 255] -> [0.0, 1.0].
			images = images.astype(np.float32)
			images = np.multiply(images, 1.0 / 255.0)
		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0
	@property
	def images(self):
		return self._images
	@property
	def labels(self):
		return self._labels
	@property
	def num_examples(self):
		return self._num_examples
	@property
	def epochs_completed(self):
		return self._epochs_completed
	def next_batch(self, batch_size, fake_data=False):
		"""Return the next `batch_size` examples from this data set."""
		if fake_data:
			fake_image = [1.0 for _ in xrange(784)]
			fake_label = 0
			return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]



def read_data_sets(train_dir, fake_data=False, one_hot=False):
	class DataSets(object):
		pass
	data_sets = DataSets()
	if fake_data:
		data_sets.train = DataSet([], [], fake_data=True)
		data_sets.validation = DataSet([], [], fake_data=True)
		data_sets.test = DataSet([], [], fake_data=True)
		return data_sets

	print("load: 1")
	data1, labels1 = loadData('./cifar-10-batches-py/data_batch_1')
	print("load: 2")
	data2, labels2 = loadData('./cifar-10-batches-py/data_batch_2')
	print("load: 3")		
	data3, labels3 = loadData('./cifar-10-batches-py/data_batch_3')
	print("load: 4")
	data4, labels4 = loadData('./cifar-10-batches-py/data_batch_4')
	print("load: 5")
	data5, labels5 = loadData('./cifar-10-batches-py/data_batch_5')
	print("Stacking")
	data = np.vstack((data1,data2,data3,data4,data5))

	labels = labels1
	labels = np.append(labels, labels2)
	labels = np.append(labels, labels3)
	labels = np.append(labels, labels4)
	labels = np.append(labels, labels5)
	#labels = np.vstack((labels1,labels2,labels3,labels4,labels5))

	#print('loadData2')
	#data = loadData2('cif.csv')

	print("Turning Gray")
	print(data.shape)
	count = 0
	dataGray = []
	for d in data:
		if count % 5000 == 0 :
			print(count)
		dataGray.append(conv2Gray(d))
		count+=1
	data = np.array(dataGray)


	VALIDATION_SIZE = 1000
	validation_images = data[:VALIDATION_SIZE]
	validation_labels = labels[:VALIDATION_SIZE]
	train_images = data[VALIDATION_SIZE:]
	train_labels = labels[VALIDATION_SIZE:]
	data_sets.train = DataSet(train_images, train_labels)
	data_sets.validation = DataSet(validation_images, validation_labels)
	#data_sets.test = DataSet(test_images, test_labels)
	return data_sets
