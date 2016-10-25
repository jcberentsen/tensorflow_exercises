""" Simple Convolutional neural network for CIFAR-10 Classification
    with regularization
    save and restore best model
    Save summary for tensorboard
    Author: Dario Cazzani & Christian Berentsen
"""
import numpy as np
import tensorflow as tf
import sys, os
import cPickle
import matplotlib.pyplot as plt
import glob
# add folder where settings.py is present
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import settings
import itertools
from sklearn import preprocessing
import time

# Helper functions
from cifar_utils import *

def load_data(folder_name):
	train_batches_list = glob.glob(folder_name + '/data_batch*')
	test_batches_list = glob.glob(folder_name + '/test_batch*')

	all_image_batches_train, all_labels_batches_train = zip(*map(load_batch_details, train_batches_list))
	all_images_train = list(itertools.chain(*all_image_batches_train))
	all_labels_train = list(itertools.chain(*all_labels_batches_train))
	assert(len(all_labels_train) == len(all_images_train))

	all_image_batches_test, all_labels_batches_test = zip(*map(load_batch_details, test_batches_list))
	all_images_test = list(itertools.chain(*all_image_batches_test))
	all_labels_test = list(itertools.chain(*all_labels_batches_test))
	assert(len(all_labels_test) == len(all_images_test))

	return all_images_train, all_labels_train, all_images_test, all_labels_test

def preprocess(data):
	# perform per image whitening
	return map(tf.image.per_image_whitening, data)

""" A simple data iterator """
def data_iterator(data, labels, batch_size):
  N = data.shape[0] 
  batch_idx = 0
  while True:
    for batch_idx in range(0, N, batch_size):
      data_batch = data[batch_idx:batch_idx+batch_size]
      labels_batch = labels[batch_idx:batch_idx+batch_size]
      yield data_batch, labels_batch


	
if __name__ == '__main__':
	folder_name = '/home/dario/Downloads/cifar-10-batches-py'
	X_train, y_train, X_test, y_test = load_data(folder_name)
	now = time.time()
	X_train = preprocess(X_train)
	print time.time() - now
	#plt.imshow(X_train[3])
	#plt.show()
	"""
	reshaped = np.reshape(data2, (32,32,3), order='A')
	plt.imshow(reshaped)
	plt.show()
	"""