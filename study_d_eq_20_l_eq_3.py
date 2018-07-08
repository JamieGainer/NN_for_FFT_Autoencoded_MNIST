import numpy as np
import os
import pickle
import sys
import tensorflow as tf

sys.path.append('../Linear_Autoencoder_and_FFT_for_MNIST/autoencoder_fft_mnist/')
import fft_autoencoder
import nn

tf.set_random_seed(1)
np.random.seed(1)

nn_arch = [20, 14, 10]
run_list = [0, 10, 20, 5, 15, 3, 8, 13, 18, 2, 7, 12, 17, 1, 6, 11, 16, 4, 9, 14, 19]
num_steps = 1000000
batch_size = 16

if 'data_pickles' not in os.listdir():
	os.mkdir('data_pickles')

mnist = fft_autoencoder.get_mnist_data_and_add_autoencodings({
	'hybrid_autoencoder': [(20 - i, i) for i in run_list]})

for i in run_list:
	dim_tuple = (20 - i, i)
	run_dict = nn.create_and_train_softmax_NN(nn_arch, mnist.train, 
		"scaled_hybrid_autoencoder", dim_tuple, num_steps, batch_size)
	with open('data_pickles/run-' + str(i) + '.pickle', 'wb') as pickle_file:
		pickle.dump(run_dict, pickle_file)
