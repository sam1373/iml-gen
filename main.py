from conv_autoencoder import ConvolutionalAutoencoder
from keras.datasets import mnist
import numpy as np

import matplotlib.pyplot as plt

from sklearn import mixture

from utility import *

"""
Implicit Maximum Likelihood generative algorithm
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

imgSize = (32, 32)
x_train = resizeImages(x_train, imgSize)
x_test = resizeImages(x_test, imgSize)

w, h = x_train.shape[-2:]

x_train = np.reshape(x_train, (len(x_train), w, h, 1))  
x_test = np.reshape(x_test, (len(x_test), w, h, 1))

model = ConvolutionalAutoencoder(w, h, [(32, 3, 3), (16, 3, 3), (8, 3, 3), (8, 3, 3)], [20])
model.autoencoder.summary()

epochs = 100
batch_size = 50
model.train(x_train, x_test, epochs, batch_size)

encoded_imgs = model.encode(x_test)
decoded_imgs = model.decode(encoded_imgs)

displayImageTable2(x_test, decoded_imgs)

X = model.encode(x_train)
n = X.shape[0]
imgShape = X.shape[1:]
X = np.reshape(X, [n, -1])

gmm = mixture.GaussianMixture(n_components=30, tol=1e-20, max_iter=3000, n_init=1, covariance_type='full', verbose=2).fit(X)

print("iterations performed:", gmm.n_iter_)

n_gen = 25

gen = gmm.sample(n_samples=n_gen)[0]
gen = np.reshape(gen, (-1,) + imgShape)
gen_decode = model.decode(gen)

displayImageTable(gen_decode, n_display=n_gen)