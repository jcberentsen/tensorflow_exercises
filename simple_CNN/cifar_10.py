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

	return np.asarray(all_images_train), np.asarray(all_labels_train), np.asarray(all_images_test), np.asarray(all_labels_test)

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
	
	# checkpoint file
	checkpoint_file = settings.PROJECT_DIR + '/simple_CNN/best_model.chk'
	# hyperparams
	batch_size = 64
	input_size = 32
	lr = 0.001
	epochs = 30001
	lmbda = 3e-3  
	num_channels = 3
	num_classes = 10

	# load data
	folder_name = '/home/dario/Downloads/cifar-10-batches-py'
	X_train, y_train, X_test, y_test = load_data(folder_name)
	data_set = data_iterator(X_train, y_train, batch_size)

	# skip preprocess for now
	#X_train = preprocess(X_train)

	# Create model
	with tf.name_scope("input") as scope:
		x = tf.placeholder(tf.float32, shape=(None, input_size, input_size, num_channels))

	with tf.name_scope("labels") as scope:
		y_ = tf.placeholder(tf.float32, shape=(None, 1))

	"""
	with tf.name_scope("conv1") as scope:
		filter1 = tf.Variable(tf.truncated_normal([2, 2, 3, 16]), name='filter1')
		conv1_out = tf.nn.conv2d(x, filter1, [1, 1, 1, 1], padding='SAME')
		with tf.name_scope("relu1") as scope:
			conv1_relu = tf.nn.relu(conv1_out)
	"""
	with tf.name_scope("conv1_flattened") as scope:
		conv1_flattened = tf.reshape(x, [-1, 32*32*3])

	with tf.name_scope("fully_connected1") as scope:
		W1 = tf.Variable(tf.random_uniform([32*32*3, num_classes]))
		b1 = tf.Variable(tf.zeros([num_classes]))
		y = tf.matmul(conv1_flattened, W1) + b1
		
	with tf.name_scope("loss") as scope:
		# Define loss - cross_entropy
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))

	# Create a summary to monitor the loss
	tf.scalar_summary("loss", loss)

	# define optimizer
	optimizer = tf.train.GradientDescentOptimizer(lr)
	train = optimizer.minimize(loss)

	# Before starting, initialize the variables.  We will 'run' this first.
	init = tf.initialize_all_variables()

	# Merge all summaries into a single operator
	merged_summary_op = tf.merge_all_summaries()

	# Launch the graph.
	with tf.Session() as sess:
		sess.run(init)
		saver = tf.train.Saver()

		# Set the logs writer to file logs.log
		summary_writer = tf.train.SummaryWriter(settings.PROJECT_DIR + '/simple_CNN/logs', graph_def=sess.graph_def)

		best_loss = np.inf
		# Train
		for step in range(epochs):
			batch = data_set.next()
			_, train_loss = sess.run([train, loss], feed_dict={x: batch[0], y_: batch[1]})
			if step % 200 == 0:
				# Write logs for each iteration
				summary_str = sess.run(merged_summary_op, feed_dict={x: batch[0], y_: batch[1]})
				summary_writer.add_summary(summary_str, step)

			print 'Loss: %.3f ' %train_loss
			if best_loss > train_loss:
				if checkpoint_file is not None:
					print "Saving variables to '%s'." % checkpoint_file
					saver.save(sess, checkpoint_file)
					best_loss = train_loss

"""
    # Test trained model
    correct_prediction = tf.nn.in_top_k(y, tf.argmax(y_, 1), 3)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict={x: data, y_: labels})
    print 'Accuracy with latest model: %.5f' %acc
    
    print "Loading variables from '%s'." % checkpoint_file
    saver.restore(sess, checkpoint_file)  
    correct_prediction = tf.nn.in_top_k(y, tf.argmax(y_, 1), 3)     
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict={x: data, y_: labels})
    print 'Accuracy with best model: %.5f' %acc

"""

























	



