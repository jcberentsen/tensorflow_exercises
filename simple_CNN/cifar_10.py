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

	return all_labels_train, all_labels_train, all_images_test, all_labels_test

def load_batch(filename):
	fo = open(filename, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

def extract_plane(raw, index):
	data = np.asarray(raw[index:index+1024], dtype=np.uint8)
	return np.reshape(data, (32, 32))

def shape_image(raw):
	# raw = np.asarray(batch_item, dtype=np.uint8)
	red = extract_plane(raw,0)
	green = extract_plane(raw, 1024)
	blue = extract_plane(raw, 2048)
	return np.dstack((red, green, blue))

def	load_batch_details(batch_filename):
	batch = load_batch(batch_filename)
	return map(shape_image, batch['data']), batch['labels']
	
if __name__ == '__main__':
	folder_name = '/home/dario/Downloads/cifar-10-batches-py'
	X_train, y_train, X_test, y_test = load_data(folder_name)
#	plt.imshow(list(all_images)[3])
#	plt.show()
	"""
	reshaped = np.reshape(data2, (32,32,3), order='A')
	plt.imshow(reshaped)
	plt.show()
	"""