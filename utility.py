from skimage.transform import resize

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from scipy import linalg

import itertools

def resizeImages(X, imgSize):
    X_resized = np.zeros([X.shape[0], imgSize[0], imgSize[1]])
    for i in range(X.shape[0]):
        X_resized[i] = resize(X[i], imgSize)
    return X_resized

def displayImageTable(imgArray, n_display=10, tblW=8, tblH=8):
    plt.figure(figsize=(5, 5))
    #w, h = imgArray.shape[-2:]
    for i in range(1,n_display+1):
        ax = plt.subplot(tblW, tblH, i)
        plt.imshow(imgArray[i - 1].squeeze())
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def displayImageTable2(imgArray, imgArray2, n_display=10, tblW=4):
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


color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, title, index=0, plotCircles=1, plots=1):
    splot = plt.subplot(plots, 1, 1 + index)
    
    xMean = X[:, 0].mean()
    yMean = X[:, 1].mean()

    for i in range(len(means)):
        means[i] -= [xMean, yMean]

    X[:, 0] -= xMean
    X[:, 1] -= yMean
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        if plotCircles:
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)