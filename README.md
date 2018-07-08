# NN_for_FFT_Autoencoded_MNIST
Exploring Autoencoded and FFT-ed MNIST Data with Neural Networks.

This repo contains code and data from my explorations of
1. Using autoencoders to reduce the feature space of images to make training neural networks more practical.
2. Using the Fast Fourier Transform (FFT) of images, rather than the images themselves, and autoencoding the FFT
of images.
3.  Mixing features from the autoencoded images and the autoencoded FFT images.

In these explorations, I am using the MNIST data set for simplicity and for ease of comparison with other results.

The only non-standard dependency is on the module autoencoder_fft_mnist/fft_autoencoder.py in my Linear_Autoencoder_and_FFT_for_MNIST github repo.

## Preliminary Conclusions

There are a lot of things left to try (deeper neural networks, more than 20 features, evaluating accuracy on the test set, etc.) but initial work suggests that training in feature spaces with both autoencoded image features and autoencoded FFT image features is not as effective in training either in spaces with only autoencoded image features or in spaces with
only autoencded FFT image features (see plots).

A potential explanation is that the FFT and non-FFT autotencoded feature spaces overlap strongly, in which case the
effective dimensionality of feature spaces with both features is lessened.

I will try to verify this explanation directly.  I note that mixing FFT and non-FFT features might still have use in preventing overfitting, even if it reduces the apparent effectiveness of training, as measured by, e.g., accuracy on the training set.