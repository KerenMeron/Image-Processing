# Naomi Deutsch
# 203516083
# naomid

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from scipy.signal import convolve as conv
from skimage.color import rgb2gray
from scipy.misc import imread, imshow

GRAY_REP = 1
RGB_REP = 2
MAX_RANGE = 256
KERNEL = np.array([1, 1])
VALUE_ERROR = "Representation should be 1 or 2"
KERNEL_ERROR = "Kernel needs to be odd"


def read_image(filename, representation):
    """
	Reads a given image file and converts it into a given representation.
	if representation == 1, will convert to grayscale,
	if representation == 2, will convert to RGB
	:param filename: the file we use
	:param representation: 1 for grayscale, 2 for RGB
	:return: in image
	"""

    if representation != GRAY_REP and representation != RGB_REP:
        raise ValueError(VALUE_ERROR)

    im = imread(filename) / MAX_RANGE
    if representation == GRAY_REP and len(im.shape) != 2:
        im = rgb2gray(im)
    im = im.astype(np.float32)
    return im


def calculate_kernel(kernel_size):
    """
    A helper function to calculate gaussian kernel
    :param kernel_size: the size of the matrix that represent the kernel.
                        must be odd integer.
    :return: a squares matrix that represent the gaussian kernel
    """
    if kernel_size % 2 == 0:
        raise ValueError(KERNEL_ERROR)

    kernel = np.array([1]).reshape(1, 1).astype(np.float32, copy=False)

    for i in range(kernel_size - 1):
        kernel = conv(KERNEL.reshape(1, 2), kernel)

    g_kernel = conv(kernel.transpose(), kernel)
    g_kernel /= np.sum(g_kernel)
    return g_kernel


def blur_spatial(im, kernel_size):
    """
    A function that performs image blurring using 2D convolution between the image f and a gaussian
    kernel g.
    :param im: a grayscale image to be blurred of dtype float32
    :param kernel_size: the size of the gaussian kernel in each dimension. must be odd integer.
    :return:  blurry grayscale image of dtype float32
    """
    g_k = calculate_kernel(kernel_size)
    blur = convolve(convolve(im, g_k), np.transpose(g_k))
    # blur = conv(im, g_k, mode="same")

    return blur.astype(np.float32)


def calculate_filter_vec(filter_size):
    """
    A helper function to calculate gaussian kernel
    :param filter_size: the size of the array that represent the kernel.
                        must be odd integer.
    :return: 1D-row of size filter_size
    """
    if filter_size % 2 == 0:
        raise ValueError(KERNEL_ERROR)

    kernel = np.array([1]).astype(np.float32, copy=False)

    for i in range(filter_size - 1):
        kernel = conv(KERNEL, kernel)

    kernel /= np.sum(kernel)
    return kernel.reshape(filter_size, 1)


def reduce(im, filter_vec):
    """
    Helper function that reduces the Gaussian image. Takes every second index and every second row.
    :param im: a Gaussian image.
    :param filter_vec: 1D-row of size filter_size (given in build_laplacian_pyramid() as an argument)
    :return: reduce image
    """
    conv_im = convolve(convolve(im, filter_vec, mode='mirror'), np.transpose(filter_vec), mode='mirror')
    reduced_im = conv_im[::2, ::2]
    return reduced_im


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
     A function that construct a Gaussian pyramid
    :param im: a grayscale image with double values in [0,1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
    :return: pyr: the pyramid - a standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image
            filter_vec: 1D-row of size filter_size used for the pyramid construction.
    """
    pyr = []
    filter_vec = calculate_filter_vec(filter_size).reshape(1, filter_size)

    # adding the origin image
    pyr.append(im.astype(np.float32))
    for i in range(max_levels - 1):
        conv_im = reduce(pyr[i], filter_vec)
        if (conv_im.shape[0] < 16) or (conv_im.shape[1] < 16): #or (conv_im.shape[0] % 2 != 0) or (conv_im.shape[1] % 2 != 0):
            break

        pyr.append(conv_im)
    return pyr, filter_vec

def zero_padding(im):
    """
    Adding zero to each second row and each second index
    :param im: image
    :return: zero padded image
    """
    row = np.arange(im.shape[0]) + 1
    col = np.arange(im.shape[1]) + 1
    padded_row = np.insert(im, row, 0, axis=0)
    padded_im = np.insert(padded_row, col, 0, axis=1)

    return padded_im


def expand(im, filter_vec):
    """
    Helper function that expands the Gaussian image to the size of the image in the next level.
    :param im: a Gaussian image.
    :param filter_vec: 1D-row of size filter_size (given in build_laplacian_pyramid() as an argument)
    :return: expand image
    """
    pad = zero_padding(im)
    expanded_im = convolve(convolve(pad, filter_vec, mode='mirror'), np.transpose(filter_vec), mode='mirror')
    return expanded_im


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
     A function that construct a Laplacian pyramid. The function uses build_gaussian_pyramid() since each phase in the
     Laplacian pyramid is a the gaussian[i] - gaussian[i-1].
    :param im: a grayscale image with double values in [0,1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Laplacian filter (an odd scalar that represents a squared filter)
    :return: pyr: the pyramid - a standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image
            filter_vec: 1D-row of size filter_size used for the pyramid construction.
    """
    g_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    for i in range(len(g_pyr) - 1):
        expend_im = expand(g_pyr[i + 1], 2 * filter_vec)
        pyr.append(g_pyr[i] - expend_im)

    # adding the last gaussian image
    pyr.append(g_pyr[len(g_pyr) - 1])
    return pyr, 2 * filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    A function that reconstruct the image from its Laplacian Pyramid
    :param lpyr: Laplacian pyramid
    :param filter_vec: 1D-row of size filter_size
    :param coeff: is a vector. The vector size is the same as the number of levels in the pyramid lpyr.
    :return: the reconstructed image
    """
    expanded_im = lpyr[len(lpyr) - 1].astype(np.float32)
    for i in range(len(lpyr) - 1, 0, -1):
        expanded_im = expand(expanded_im, filter_vec) * coeff[i] + lpyr[i - 1]
    return expanded_im


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Implement pyramid blending
    :param im1: grayscale images to be blended.
    :param im2: grayscale images to be blended.
    :param mask: is a boolean (i.e. dtype == np.bool) mask containing True and False representing which parts
                of im1 and im2 should appear in the resulting im_blend
    :param max_levels: max level of the pyramid
    :param filter_size_im: is the size of the Gaussian filter
    :param filter_size_mask: is the size of the Gaussian filter
    :return: blended image
    """
    L_1, F_V1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L_2, F_V2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    G_m, G_F_V = build_gaussian_pyramid(mask.astype(np.float32), max_levels, filter_size_mask)
    L_out = []

    for i in range(len(G_m)):
        L_out.append((G_m[i] * L_1[i]) + ((1 - G_m[i]) * L_2[i]))

    im_blend = laplacian_to_image(L_out, F_V1, np.ones(len(L_out)))
    im_blend = np.clip(im_blend, 0, 1)

    return im_blend
