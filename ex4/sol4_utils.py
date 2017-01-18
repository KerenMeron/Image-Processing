##############################################################################
# FILE: sol4_utils.py
# WRITER: Keren Meron, keren.meron, 200039626
# EXERCISE: Image Processing ex4 utils 2016-2017
# DESCRIPTION: utilities for sol4.py such as pyramid blending, pyramid building, blur, etc.
##############################################################################

import numpy as np
import scipy.signal
from scipy.misc import imread, imsave
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os
from scipy import ndimage


EXPAND_FACTOR = 2
SAMPLE_JUMP = 2
NORMALIZE_FACTOR = 255
DEFAULT = 333
FILE_ERROR_MSG = "Error in file."
GRAY_SCALE = 1
RGB_SIZE = 3
BASE_KERNEL = [1, 1]
DERIVE_VEC = np.array([[1], [0], [-1]])


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


def get_filter_vec(kernel_size):
    """
    Calculate a gaussian kernel (according to pascal's triangle)
    :param kernel_size: degree of kernel
    :return: filter vec, the kernel
    """
    kernel_base = np.array(BASE_KERNEL)
    kernel = np.array(BASE_KERNEL)
    for i in range(kernel_size - 2):
        kernel = np.convolve(kernel, kernel_base)
    filter_vec = kernel / np.sum(kernel)
    return np.reshape(filter_vec, (1, filter_vec.size))


def reduce(im, kernel):
    """
    Reduces a given image to twice its size
    :param im: grayscale image with double values in [0,1]
    :return: expanded im, same type as im
    """
    blurred = blur_spatial_kernel(im, kernel)
    return blurred[::SAMPLE_JUMP, ::SAMPLE_JUMP]


def blur_spatial(im, kernel_size):
    """
    Performs image blurring using 2d convolution between a given image and a gaussian kernel
    :param im: input image to be blurred, grayscale float32
    :param kernel_size: size of the gaussian kernel
    :return: blurry image, grayscale float32
    """
    kernel = get_Gauss_Kernel(kernel_size)
    blur_im = scipy.signal.convolve2d(im, kernel, mode='same', boundary='fill')
    return blur_im


def get_Gauss_Kernel(kernel_size):
    """
    Calculate a gaussian kernel (according to pascal's triangle)
    :param kernel_size: degree of kernel
    :return: the gaussian kernel
    """
    kernel_base = np.array([1, 1])
    kernel = np.array([1, 1])
    for i in range(kernel_size - 2):
        kernel = np.convolve(kernel, kernel_base)
    full_kernel = np.multiply.outer(np.transpose(kernel), np.array(kernel)).astype(np.float32)
    return full_kernel / np.sum(full_kernel)


def conv_der(im):
    """
    Computes the magnitude of a given image's derivatives.
    :param im: grayscale image of float32
    :return: magnitude: grayscale image of float32
    """
    row_deriv = scipy.signal.convolve2d(im, DERIVE_VEC, mode='same', boundary='symm')
    col_deriv = scipy.signal.convolve2d(im, np.array(np.transpose(DERIVE_VEC)), mode='same', boundary='symm')
    return np.sqrt(np.square(col_deriv) + np.square(row_deriv))



def blur_spatial_kernel(im, kernel):
    """
    Blur a given image by convolving with a given kernel
    :param im: grayscale image with double values in [0,1]
    :param kernel: 1D-row used for pyramid construction, normalized
    :return: blurred image
    """
    rows = ndimage.filters.convolve(im, kernel, mode='wrap').astype(np.float32)
    return ndimage.filters.convolve(rows, kernel.T, mode='wrap').astype(np.float32)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Constructs a guassian pyramid of a given image.
    :param im: grayscale image with double values in [0,1]
    :param max_levels: maximal number of levels in resulting pyramid
    :param filter_size: size of guassian filter (odd scalar)
    :return: pyr:           python array of grayscale images with max length of max_levels
             filter_vec:    1D-row used for pyramid construction, normalized
    """
    filter_vec = get_filter_vec(filter_size)
    pyr = [im]
    reduced_im = im
    for i in range(max_levels - 1):
        reduced_im = reduce(reduced_im, filter_vec)
        pyr.append(reduced_im)
    return pyr, filter_vec


def pad_zeros(im, dest_size):
    """
    Pad image with zeros in uneven indices (between rows and columns, excluding boundaries)
    :param im: grayscale image with double values in [0,1]
    :return: padded im, same type as im
    """
    padded = np.insert(im, slice(1, None), 0, axis=1)
    padded = np.insert(padded, slice(1, None), 0, axis=0)

    if padded.shape[1] != dest_size[1]:
        indices_col = np.zeros((padded.shape[0], 1)).astype(np.float32)
        padded = np.append(padded, indices_col, axis=1)
    if padded.shape[0] != dest_size[0]:
        indices_row = np.zeros((1, padded.shape[1])).astype(np.float32)
        padded = np.append(padded, indices_row, axis=0)
    return padded


def expand(im, kernel, dest_size):
    """
    Expands a given image to twice its size
    :param im: grayscale image with double values in [0,1]
    :return: expanded im, same type as im
    """
    padded_im = pad_zeros(im, dest_size)
    return blur_spatial_kernel(padded_im, kernel * EXPAND_FACTOR).astype(np.float32)


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Constructs a laplacian pyramid of a given image.
    :param im: grayscale image with double values in [0,1]
    :param max_levels: maximal number of levels in resulting pyramid
    :param filter_size: size of guassian filter (odd scalar)
    :return: pyr:           python array of grayscale images with max length of max_levels
             filter_vec:    1D-row used for pyramid construction, normalized
    """
    filter_vec = get_filter_vec(filter_size)
    guassian_pyr = build_gaussian_pyramid(im, max_levels, filter_size)[0]
    pyr = []
    for i in range(max_levels-1):
        pyr.append(guassian_pyr[i] - expand(guassian_pyr[i + 1], filter_vec, guassian_pyr[i].shape))
    pyr.append(guassian_pyr[max_levels-1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Reconstructs an image from its laplacian pyramid, by summing all laplacians
    :param lpyr: laplacian pyramid (array of images)
    :param filter_vec: vector used to create laplacians
    :param coeff: vector of size number of levels in pyramid
    :return: reconstructed image
    """
    lpyr = np.multiply(lpyr, coeff)
    end = len(lpyr) - 1
    img = lpyr[end]
    for i in range(end - 1, -1, -1):
        img = expand(img, filter_vec, lpyr[i].shape) + lpyr[i]
    return img.astype(np.float32)


def canvas_size(levels, width):
    """
    Calculate size of canvas needed to display a given number of images of decreasing sizes.
    Sizes of images decrease by 2
    :param levels: number of images
    :param first_size: shape (length, width) of biggest image
    :return: (length, width) in pixels
    """
    sum_width = width
    for i in range(levels - 1):
        width = width / 2
        sum_width += width
    rounded = np.floor(sum_width)
    if sum_width == rounded:
        return sum_width
    return sum_width + 1



def fix_levels(max_levels, im_shape):
    """
    Checks if pyramid blending can be performed with given images for a certain max levels.
    If not possible, finds maximal allowed levels.
    :return: correct max levels (int)
    """

    def legal_multiple(levels, axis_size):
        if axis_size < 16:
            return 1

        while axis_size % (2 ** (levels - 1)) != 0:
            levels -= 1

        real_levels = 1
        while axis_size > 16 and real_levels < levels:
            axis_size /= 2
            real_levels += 1
        return real_levels

    return min(legal_multiple(max_levels, im_shape[0]), legal_multiple(max_levels, im_shape[1]))



def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Blends two given images by building their laplacian pyramids, joining them according to a given mask to one
    pyramid, and summing that pyramid.
    Assumes both images and the mask are all in the same size.
    :param im1, im2: two grayscale images to be blended
    :param mask: np.bool mask containing True(1) and False(0) representing which parts of im1 and im2 should appear
    :param max_levels: levels in pyramids
    :param filter_size_im: size of gaussian filter (odd scalar) for filter kernel of images' pyramids
    :param filter_size_mask: size of gaussian filter (odd scalar) for filter kernel of mask's pyramid
    :return: im_blend: blended image
    """
    # max_levels = fix_levels(max_levels, im1.shape)
    laplac_im1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    laplac_im2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    mask_pyr = build_gaussian_pyramid(mask.astype(np.float32), max_levels, filter_size_mask)[0]
    laplacian_out = list()
    for i in range(max_levels):
        laplacian_out.append(np.multiply(mask_pyr[i], laplac_im1[i]) + np.multiply(1 - mask_pyr[i], laplac_im2[i]))
    coeff = np.ones(max_levels)
    im_blend = laplacian_to_image(laplacian_out, filter_vec, coeff).astype(np.float32)
    return np.clip(im_blend, 0, 1)



def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

