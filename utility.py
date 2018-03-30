from skimage.transform import resize

import matplotlib.pyplot as plt

import numpy as np

def resizeImages(X, imgSize):
    X_resized = np.zeros([X.shape[0], imgSize[0], imgSize[1]])
    for i in range(X.shape[0]):
        X_resized[i] = resize(X[i], imgSize)
    return X_resized

def displayImageTable(imgArray, n_display=10, tblW=5, tblH=5):
    plt.figure(figsize=(5, 5))
    #w, h = imgArray.shape[-2:]
    for i in range(1,n_display+1):
        ax = plt.subplot(tblW, tblH, i)
        plt.imshow(imgArray[i - 1].squeeze())
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def displayImageTable2(imgArray, imgArray2, n_display=5, tblW=4):
    plt.figure(figsize=(5, 5))
    #w, h = imgArray.shape[-2:]
    for i in range(1, n_display + 1):
        ax = plt.subplot(2, n_display, i)
        plt.imshow(imgArray[i].squeeze())
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n_display, i + n_display)
        plt.imshow(imgArray2[i].squeeze())
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()