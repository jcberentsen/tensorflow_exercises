""" 1 hidden layer neural network for regression
    Author: Dario Cazzani
"""
import numpy as np
import tensorflow as tf
from tflearn.data_utils import load_csv, to_categorical

def load_data(filename):
  data, labels = load_csv(filename, target_column=8, categorical_labels=False)
  num_classes = len(set(labels))

  return data, labels, num_classes

def preprocess(data, labels):
  for i in range(len(data)):
    if data[i][0] == 'M':
      data[i][0] = 0.
    elif data[i][0] == 'F':
      data[i][0] = 1.
    else:
      data[i][0] = 2.
  labels = np.asarray(labels, dtype=np.float32)
  labels = np.reshape(labels, (labels.shape[0], 1))
  return np.asarray(data, dtype=np.float32), labels


""" A simple data iterator """
def data_iterator(data, labels, batch_size):
  N = data.shape[0] 
  batch_idx = 0
  while True:
    for batch_idx in range(0, N, batch_size):
      data_batch = data[batch_idx:batch_idx+batch_size]
      labels_batch = labels[batch_idx:batch_idx+batch_size]
      yield data_batch, labels_batch

def create_model(input_size):
  x = tf.placeholder(tf.float32, shape=(None, input_size))
  y_ = tf.placeholder(tf.float32, shape=(None, 1))
  W1 = tf.Variable(tf.random_uniform([input_size, hidden1], -1.0, 1.0))
  b1 = tf.Variable(tf.zeros([hidden1]))

  W2 = tf.Variable(tf.random_uniform([hidden1, hidden2], -1.0, 1.0))
  b2 = tf.Variable(tf.zeros([hidden2]))
  h1 = tf.nn.tanh(tf.matmul(x, W1) + b1)

  W3 = tf.Variable(tf.random_uniform([hidden2, 1], -1.0, 1.0))
  b3 = tf.Variable(tf.zeros([1]))
  h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)

  y = tf.matmul(h2, W3) + b3

  return x, y, y_

if __name__ == '__main__':
  
  # hyperparams
  batch_size = 64
  input_size = 8
  hidden1 = 16
  hidden2 = 16
  lr = 0.03
  epochs = 10001

  # load data
  data, labels, num_classes = load_data('/home/dario/Dev/tensorflow_studygroup/datasets/abalone.csv')
  data, labels = preprocess(data, labels)
  data_set = data_iterator(data, labels, batch_size)

  # Create model
  x, y, y_ = create_model(input_size)

  # Define loss and optimizer
  loss = tf.reduce_mean(tf.square(y - y_))
  optimizer = tf.train.GradientDescentOptimizer(lr)
  train = optimizer.minimize(loss)

  # Before starting, initialize the variables.  We will 'run' this first.
  init = tf.initialize_all_variables()

  # Launch the graph.
  sess = tf.Session()
  sess.run(init)

  # Test
  test1 = np.asarray((1.,0.53,0.415,0.15,0.7775,0.237,0.1415,0.33)).reshape((1,input_size))    
  correct1 = 20.

  # Train
  for step in range(epochs):
    batch = data_set.next()
    _, train_loss = sess.run([train, loss], feed_dict={x: batch[0], y_: batch[1]})
    if step % 200 == 0:
      prediction = sess.run(y, feed_dict={x : test1})
      print 'Loss: %.3f, Prediction: %.2f, Correct: %.2f' %(train_loss, prediction, correct1)