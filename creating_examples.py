from matplotlib import pyplot as plt
from PIL import Image
import numpy

import numpy as np
import scipy.signal
from scipy.misc import imread, imsave
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage import color


EXPAND_FACTOR = 2
SAMPLE_JUMP = 2
NORMALIZE_FACTOR = 255
DEFAULT = 333
FILE_ERROR_MSG = "Error in file."
GRAY_SCALE = 1
RGB_SIZE = 3
BASE_KERNEL = [1, 1]


def isRGB(img):
    """
    Checks whether given image is in color RGB or grayscale
    :param img: given image, matrix of pixels
    :return: True if RGB, False if grayscale
    """
    return len(img.shape) == RGB_SIZE and img.shape[-1] >= RGB_SIZE


def read_image(filename, representation=DEFAULT):
    """
    Reads image from a given filename and turns into a given representation
    :param filename: name of file containing image
    :param representation: e.g. 1 for grayscale, 2 for RGB, 333 for default (no change)
    :return: matrix representing the image, with normalized intensities in [0,1]
    :raise: SystemExit if file not found / error in file.
    """
    try:
        image = imread(filename).astype(np.float32) / NORMALIZE_FACTOR
    except:
        print(FILE_ERROR_MSG)
        raise SystemExit
    if representation == GRAY_SCALE and isRGB(image):
        image = rgb2gray(image)
    elif len(image.shape) != RGB_SIZE:
        image = rgb2gray(image)     #if grey turn into 2d array
    elif image.shape[2] > RGB_SIZE:  #png
        image = image[:,:,:3]
    return image

# def read_image(filename, representation):
#     im = imread(filename)
#     if representation == 1 and im.ndim == 3 and im.shape[2] == 3:
#         im = color.rgb2gray(im)
#     if im.dtype == np.uint8:
#         im = im.astype(np.float32) / 255.0
#     return im


filename = "examples/spa/mask_wind.png"
image = read_image(filename, GRAY_SCALE)
plt.imshow(image, plt.cm.gray)
plt.show()
for i in range(len(image)):
    for j in range(len(image[0])):
        if image[i][j] == 0:
            image[i][j] = 1
        else:
            image[i][j] = 0
plt.imshow(image, plt.cm.gray)
plt.show()
imsave('examples/spa/mask_wind_f.png', image)

