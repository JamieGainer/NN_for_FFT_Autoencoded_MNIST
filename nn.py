"""
Header
"""

import numpy as np
import sys
import tensorflow as tf
import time


def init_weights(shape, stddev = 0.1):
	init_random_dist = tf.truncated_normal(shape, stddev=stddev)
	return tf.Variable(init_random_dist)


def next_batch(samples, labels, batch_size):
	indices = np.random.randint(0, len(samples), size = batch_size)
	return samples[indices], labels[indices]


def np_sigmoid(x):
	return 1./(1 + np.exp(-x))


def np_nn(new_samples, W_list, b_list, activation_function):
	X = new_samples
	a_list = [X]
	z_list = []

	for W, b in zip(W_list, b_list):
		a = a_list[-1]
		z = a.dot(W) + b
		z_list.append(z)
		a_list.append(activation_function(z))

	return z_list[-1]


def create_and_train_softmax_NN(neuron_layer_sizes, mnist_data_set,
								auto_enocder_type, dim_key, num_steps,
								batch_size,
								activation_function = tf.sigmoid,
								optimizer = tf.train.AdamOptimizer(),
								saved_W_list = None,
								saved_b_list = None, 
								print_status = True):
	
	start_time = time.time()
	
	assert neuron_layer_sizes[-1] == 10
	
	X = tf.placeholder(tf.float32,shape=[None, neuron_layer_sizes[0]])
	y_true = tf.placeholder(tf.float32,shape=[None, neuron_layer_sizes[-1]])
		
	if saved_W_list:
		W_list = [tf.Variable(W) for W in saved_W_list]
	else:
		W_list = [init_weights((n_in, n_out)) for n_in, n_out in 
		zip(neuron_layer_sizes[:-1], neuron_layer_sizes[1:])]
	
	if saved_b_list:
		b_list = [tf.Variable(b) for b in saved_b_list]
	else:
		b_list = [init_weights((n,)) for n in neuron_layer_sizes[1:]]

	a_list = [X]
	z_list = []
	
	for W, b in zip(W_list, b_list):
		a = a_list[-1]
		z = tf.matmul(a, W) + b
		z_list.append(z)
		a_list.append(activation_function(z))
		
	y_pred = z_list[-1]
	
	loss_function = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(
			labels=y_true, logits=y_pred))
	
	matches = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
	accuracy = tf.reduce_mean(tf.cast(matches,tf.float32))

	step_numbers = []
	loss_function_values = []
	accuracy_values = []
	
	train = optimizer.minimize(loss_function)    
	init = tf.global_variables_initializer()
	
	samples = getattr(mnist_data_set, auto_enocder_type)[dim_key]
	labels = mnist_data_set.labels
	
	with tf.Session() as sess:
		sess.run(init)
		
		for step in range(num_steps):

			if ((step < 10) or
				(step < 100 and step % 10 == 0) or
				(step < 1000 and step % 100 == 0) or
				(step < 10000 and step % 1000 == 0) or
				(step < 100000 and step % 10000 == 0) or
				(step % 100000 == 0) or
				step == num_steps - 1):
			
				step_numbers.append(step)
				loss_function_values.append(
					sess.run(loss_function, feed_dict = {X: samples, y_true: labels}))

				accuracy_values.append(
					sess.run(accuracy, feed_dict = {X: samples, y_true: labels}))
				if print_status:
					print('On step', step)
					
			batch_X, batch_y = next_batch(samples, labels, batch_size)
			sess.run(train, feed_dict={X: batch_X, y_true: batch_y})

		W_list_values = sess.run(W_list)
		b_list_values = sess.run(b_list)

		final_values = sess.run(y_pred, feed_dict = {X: samples})

	return {
		'step_numbers': step_numbers,
		'loss_function_values': loss_function_values,
		'accuracy_values': accuracy_values,
		'W': W_list_values,
		'b': b_list_values,
		'final_values': final_values,
		'start': start_time,
		'finish': time.time(),
		'neuron_layer_sizes': neuron_layer_sizes,
		'auto_enocder_type': auto_enocder_type,
		'dim_key': dim_key, 
		'num_steps': num_steps,
		'batch_size': batch_size}    
