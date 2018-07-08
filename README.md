# NN_for_FFT_Autoencoded_MNIST
Exploring Autoencoded and FFT-ed MNIST Data with Neural Networks.

This repo contains code and data from my explorations of
1. Using autoencoders to reduce the feature space of images to make training neural networks more practical.
2. Using the Fast Fourier Transform (FFT) of images, rather than the images themselves, and autoencoding the FFT
of images.
3.  Mixing features from the autoencoded images and the autoencoded FFT images.

In these explorations, I am using the MNIST data set for simplicity and for ease of comparison with other results.

The only non-standard dependency is on the module autoencoder_fft_mnist/fft_autoencoder.py in my Linear_Autoencoder_and_FFT_for_MNIST github repo.

