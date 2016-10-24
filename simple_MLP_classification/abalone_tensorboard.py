""" 2 hidden layers neural network for classification
    with regularization
    save and restore best model
    Save summary for tensorboard
    Author: Dario Cazzani & Christian Berentsen
"""
import numpy as np
import tensorflow as tf
from tflearn.data_utils import load_csv
import sys, os
# add folder where settings.py is present
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import settings

def load_data(filename):
  num_classes = 30
  data, labels = load_csv(filename, categorical_labels=True, n_classes=num_classes)
  return data, labels, num_classes

def preprocess(data):
  for i in range(len(data)):
    if data[i][0] == 'M':
      data[i][0] = 0.
    elif data[i][0] == 'F':
      data[i][0] = 1.
    else:
      data[i][0] = 2.
  return np.asarray(data, dtype=np.float32)

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
  checkpoint_file = settings.PROJECT_DIR + '/simple_MLP_classification/best_model.chk'
  # hyperparams
  batch_size = 64
  input_size = 8
  hidden1 = 32
  hidden2 = 16
  lr = 0.001
  epochs = 30001
  lmbda = 3e-3  

  # load data
  data, labels, num_classes = load_data(settings.PROJECT_DIR + '/datasets/abalone.csv')
  data = preprocess(data)
  data_set = data_iterator(data, labels, batch_size)

  # Create model
  with tf.name_scope("input") as scope:
    x = tf.placeholder(tf.float32, shape=(None, input_size))

  with tf.name_scope("labels") as scope:
    y_ = tf.placeholder(tf.float32, shape=(None, num_classes))

  with tf.name_scope("hidden1") as scope:
    W1 = tf.Variable(tf.random_uniform([input_size, hidden1], -1.0, 1.0), name='W1')
    b1 = tf.Variable(tf.zeros([hidden1]), name='b1')
    with tf.name_scope("relu1") as scope:
      h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

  with tf.name_scope("hidden2") as scope:
    W2 = tf.Variable(tf.random_uniform([hidden1, hidden2], -1.0, 1.0), name='W2')
    b2 = tf.Variable(tf.zeros([hidden2]), name='b2')
    with tf.name_scope("relu2") as scope:
      h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

  with tf.name_scope("output") as scope:
    W3 = tf.Variable(tf.random_uniform([hidden2, num_classes], -1.0, 1.0), name='W3')
    b3 = tf.Variable(tf.zeros([num_classes]), name='b3')
    with tf.name_scope("linear_output") as scope:
      y = tf.matmul(h2, W3) + b3

  with tf.name_scope("loss") as scope:
    # Define loss - cross_entropy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_)) 
    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) +
                    tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2))
    # Add the regularization term to the loss.
    loss += lmbda * regularizers

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
    summary_writer = tf.train.SummaryWriter(settings.PROJECT_DIR + '/simple_MLP_classification/logs', graph_def=sess.graph_def)

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


