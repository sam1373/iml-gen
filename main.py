from conv_autoencoder import ConvolutionalAutoencoder
from keras.datasets import mnist
import numpy as np

import matplotlib.pyplot as plt

from sklearn import mixture

from utility import *

import argparse

"""
Implicit Maximum Likelihood generative algorithm
"""

argparser = argparse.ArgumentParser()
argparser.add_argument('--epochs', type=int, help='number of epochs for autoencoder training', default=250)
argparser.add_argument('--batch_size', type=int, help='size of batches for autoencoder training', default=3000)
argparser.add_argument('--seed', type=int, help='random seed', default=1)
argparser.add_argument('--comp', type=int, help='gaussian mixture components', default=30)
argparser.add_argument('--inits', type=int, help='gaussian mixture fit initializations', default=20)

arguments = argparser.parse_args()

np.random.seed(arguments.seed)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x_train = x_train[y_train <= 1]
#_test = x_test[y_test <= 1]

#imgSize = (32, 32)
#x_train = resizeImages(x_train, imgSize)
#x_test = resizeImages(x_test, imgSize)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

w, h = x_train.shape[-2:]

x_train = np.reshape(x_train, (len(x_train), w, h, 1))  
x_test = np.reshape(x_test, (len(x_test), w, h, 1))

#[(32, 3, 3), (16, 3, 3), (8, 3, 3), (8, 3, 3)]
#[600, 20]
model = ConvolutionalAutoencoder(w, h, [], [300, 10])
model.autoencoder.summary()

epochs = arguments.epochs
batch_size = arguments.batch_size
model.train(x_train, x_test, epochs, batch_size)

encoded_imgs = model.encode(x_test)
decoded_imgs = model.decode(encoded_imgs)

displayImageTable2(x_test, decoded_imgs)

X = model.encode(x_train)

gmm = mixture.GaussianMixture(n_components=arguments.comp, verbose=1, n_init=arguments.inits, max_iter = 200).fit(X)

n_gen = 64

gen = gmm.sample(n_samples=n_gen)[0]
gen_decode = model.decode(gen)

displayImageTable(gen_decode, n_display=n_gen)

