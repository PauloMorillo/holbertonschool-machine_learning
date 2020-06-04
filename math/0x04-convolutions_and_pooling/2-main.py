#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_padding = __import__('2-convolve_grayscale_padding').convolve_grayscale_padding


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/animals_1.npz')
    images = dataset['data']
    print(images.shape)
    kernel = np.array([[0, 0, 0], [-1, -1, -1], [0, 0, 0]])
    images[0, :, :, 1] = 255
    images[0, :, :, 2] = 255
    images[0, :, :, 0] = 255
    images_conv = convolve_grayscale_padding(images[:,:,:,0], kernel, (2, 4))
    print(images_conv.shape)
    plt.imshow(images[0])
    plt.show()
    plt.imshow(images_conv[0])
    print(images[0])
    plt.show()