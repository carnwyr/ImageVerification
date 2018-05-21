'''
Describes CNN model and trains it  
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
from glob import glob
import os
import filters
from sklearn.utils import shuffle
import sys
import telegram_send

tf.logging.set_verbosity(tf.logging.INFO)

img_size = 64
num_channels = 3

filter_size_conv1 = 5 
num_filters_conv1 = 30

filter_size_conv2 = 3
num_filters_conv2 = 16

initial_learning_rate = 0.001
momentum = 0.5
beta = 0.0005

bias_count = 0

# load and preprocess image patches for training or testing
def load_data(classes, train_path):
	images = []
	labels = []

	print('Reading images')
	for fields in classes:
		index = classes.index(fields)
		print('Reading {} files (Index: {})'.format(fields, index))
		path = os.path.join(train_path, fields, '*')
		files = glob(path)
		for fl in files:
			image = cv2.imread(fl)
			image = image.astype(np.float32)
			image = np.multiply(image, 1.0 / 255.0)
			images.append(image)
			labels.append(index)
	images = np.array(images)
	labels = np.array(labels)

	return images, labels

def create_weights(shape):
	initializer = tf.contrib.layers.xavier_initializer()
	return tf.Variable(initializer(shape))

def create_biases(size):
	global bias_count
	bias_count += 1
	initializer = tf.contrib.layers.xavier_initializer()
	return tf.Variable(initializer((size,)),name='bias'+str(bias_count))

# separate function as the weights are initialized with SRM filters
def create_first_layer(input, num_input_channels, conv_filter_size, num_filters):
	weights = filters.get_filters_as_weights()
	biases = create_biases(num_filters)

	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='VALID')
	layer = tf.nn.bias_add(layer, biases)
	layer = tf.nn.relu(layer)

	return layer

def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
	weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
	biases = create_biases(num_filters)

	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='VALID')
	layer = tf.nn.bias_add(layer, biases)
	layer = tf.nn.relu(layer)

	return layer

def create_max_pooling_layer(input):
	layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	return layer


def cnn_model_fn(features, labels, mode):
	global bias_count
	bias_count = 0

	input_layer = tf.reshape(features["x"], [-1, img_size, img_size, num_channels])

	layer_conv1 = create_first_layer(input=input_layer, num_input_channels=num_channels, conv_filter_size=filter_size_conv1, num_filters=num_filters_conv1)
	layer_conv2 = create_convolutional_layer(input=layer_conv1, num_input_channels=num_filters_conv1, conv_filter_size=filter_size_conv1, num_filters=num_filters_conv1)
	layer_norm1 = tf.nn.lrn(layer_conv2)
	layer_pool1 = create_max_pooling_layer(input=layer_norm1)
	layer_conv3 = create_convolutional_layer(input=layer_pool1, num_input_channels=num_filters_conv1, conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2)
	layer_conv4 = create_convolutional_layer(input=layer_conv3, num_input_channels=num_filters_conv2, conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2)
	layer_conv5 = create_convolutional_layer(input=layer_conv4, num_input_channels=num_filters_conv2, conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2)
	layer_norm2 = tf.nn.lrn(layer_conv5)
	layer_pool2 = create_max_pooling_layer(input=layer_norm2)
	layer_conv6 = create_convolutional_layer(input=layer_pool2, num_input_channels=num_filters_conv2, conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2)
	layer_conv7 = create_convolutional_layer(input=layer_conv6, num_input_channels=num_filters_conv2, conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2)
	layer_conv8 = create_convolutional_layer(input=layer_conv7, num_input_channels=num_filters_conv2, conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2)
	layer_conv9 = create_convolutional_layer(input=layer_conv8, num_input_channels=num_filters_conv2, conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2)

	layer_shape = layer_conv9.get_shape().as_list()
	layer_flat = tf.reshape(layer_conv9, [-1, layer_shape[1] * layer_shape[2] * layer_shape[3]])

	dropout = tf.layers.dropout(inputs=layer_flat, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

	logits = tf.layers.dense(inputs=dropout, units=2, activation=None)

	predictions = {
			"classes": tf.argmax(input=logits, axis=1),
			"probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
			"descriptor": layer_flat
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# loss uses L2 regularization for weights
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=logits))
	variables = tf.trainable_variables()
	lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in variables if 'bias' not in v.name ])
	loss = loss + beta * lossL2
	
	# export vars for TensorBoard
	acc = tf.equal(tf.cast(predictions["classes"], tf.float32), tf.cast(labels, tf.float32))
	acc = tf.reduce_mean(tf.cast(acc, tf.float32))
	tf.summary.scalar("loss", loss)
	tf.summary.scalar("acc", acc)
	summary_hook = tf.train.SummarySaverHook(save_steps=100, output_dir='tampered_authentic_model/graph', summary_op=tf.summary.merge_all())

	if mode == tf.estimator.ModeKeys.TRAIN:
		learning_rate = tf.train.exponential_decay(initial_learning_rate, tf.train.get_global_step(), 100000, 0.9, staircase=True)
		optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[summary_hook])

	eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(argv):
	b_size = 1
	epochs = int(float(sys.argv[1]))
	classes = ['tampered', 'authentic']
	train_path = 'training_data_new'
	test_path = 'testing_data_new'

	train_data, train_labels = load_data(classes, train_path)
	train_data, train_labels = shuffle(train_data, train_labels)

	eval_data, eval_labels = load_data(classes, test_path)

	ps_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="tampered_authentic_model")

	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

	elems = train_data.shape[0]
	for it in range(epochs):
		# train the model
		train_input_fn = tf.estimator.inputs.numpy_input_fn(
				x={"x": train_data},
				y=train_labels,
				batch_size=b_size,
				num_epochs=None,
				shuffle=True)
		ps_classifier.train(
				input_fn=train_input_fn,
				steps=elems//b_size,
				hooks=[logging_hook])

		# evaluate the model and print results
		print('CASIA2 eval:')
		eval_input_fn = tf.estimator.inputs.numpy_input_fn(
				x={"x": eval_data},
				y=eval_labels,
				num_epochs=1,
				shuffle=False)
		eval_results = ps_classifier.evaluate(input_fn=eval_input_fn)
		print(eval_results)
		message=str(eval_results['accuracy'])
		telegram_send.send(messages=[message])

	telegram_send.send(messages=["done!"])

if __name__ == "__main__":
	tf.app.run()