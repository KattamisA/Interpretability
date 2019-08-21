import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image

image_dataset = ['it_{}.png'.format(200*i) for i in range(1, 3)]
# image_dataset.extend(['it_{}.png'.format(200*i) for i in range(3, 100)])
# print(image_dataset)
for i in image_dataset:

    img = cv2.imread('results/DIP_aero/DIP_output/{}'.format(i))[..., ::-1]
    # img = cv2.imread('data/F16_GT.png')[..., ::-1]
    img = np.mean(img, 2)
    # load the image data into a numpy array
    img_data = np.asarray(img)
    # perform the 2-D fast Fourier transform on the image data
    fourier = np.fft.fft2(img_data)
    # move the zero-frequency component to the center of the Fourier spectrum
    fourier = np.fft.fftshift(fourier)
    # compute the magnitudes (absolute values) of the complex numbers
    fourier = np.abs(fourier)
    # compute the common logarithm of each value to reduce the dynamic range
    fourier = np.log10(fourier)
    # find the minimum value that is a finite number
    lowest = np.nanmin(fourier[np.isfinite(fourier)])
    # find the maximum value that is a finite number
    highest = np.nanmax(fourier[np.isfinite(fourier)])
    # calculate the original contrast range
    original_range = highest - lowest
    # normalize the Fourier image data ("stretch" the contrast)
    norm_fourier = (fourier - lowest) / original_range * 255
    # convert the normalized data into an image
    norm_fourier_img = Image.fromarray(norm_fourier)
    plt.imsave('results/DIP_aero/frequencies/{}'.format(i), norm_fourier_img, format='png')
