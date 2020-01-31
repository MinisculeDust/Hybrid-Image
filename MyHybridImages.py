
import numpy as np
from MyConvolution import convolve
# version 3.3 2019.11.7-18:57


def makeGaussianKernel(sigma):

    # set the sacle of templet
    size = (int)(8 * sigma + 1) # (this implies the window is + / - 4 sigmas from the centre of the Gaussian)
    if (size % 2 == 0):
        size += 1   # size must be odd

    # templet's centre position
    center = (size - 1) / 2

    # state kernel
    kernel = np.zeros((size, size))

    kernel_sum = 0
    # gaussian calculating
    for i in range(size):
        x2 = pow(i - center, 2)
        for j in range(size):
            y2 = pow(j - center, 2)
            g = np.exp(-(x2 + y2)/(2 * sigma * sigma)) / (2 * np.pi * sigma * sigma)
            kernel[i][j] = g
            kernel_sum += kernel[i][j]

    # normalisation
    # kernel_sum2 = 0
    # for i in range(size):
    #     for j in range(size):
    #         kernel[i][j] = kernel[i][j] / kernel_sum
    #         kernel_sum2 += kernel[i][j]

    return kernel

def myHybridImages(lowImage: np.ndarray, lowSigma, highImage: np.ndarray, highSigma):

    # make kernel
    low_kernel = makeGaussianKernel(lowSigma)
    high_kernel = makeGaussianKernel(highSigma)

    # convolve low-pass pictures
    low_image = convolve(lowImage, low_kernel)

    # make high-pass picture
    high_image = (highImage - convolve(highImage, high_kernel))

    # final picture
    # the weights between and final lighting can be changed flexibly
    weight = 1
    weight2 = 1
    adjustment = 0
    hybrid_image =  high_image * weight2 + low_image * weight + adjustment
    # hybrid_image = high_image + low_image

    # randomly double check the output
    # print(hybrid_image[11][22][1])
    # print(hybrid_image[44][55][0])
    # print(hybrid_image[357][159][2])

    return hybrid_image

