##############################################################################
# FILE: sol2.py
# WRITER: Keren Meron, keren.meron, 200039626
# EXERCISE: Image Processing ex2 2016-2017
# DESCRIPTION: DFT, convolution, derivatives, blurring images
##############################################################################

import numpy as np
import scipy.signal
from scipy.misc import imread
from skimage.color import rgb2gray


DERIVE_VEC = np.array([[1], [0], [-1]]) * 0.5
GRAY_SCALE = 1
RGB_SIZE = 3
NORMALIZE_FACTOR = 255
DEFAULT = 333
FILE_ERROR_MSG = "Error in file."


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
    return image


def isRGB(img):
    """
    Checks whether given image is in color RGB or grayscale
    :param img: given image, matrix of pixels
    :return: True if RGB, False if grayscale
    """
    return len(img.shape) == RGB_SIZE and img.shape[-1] == RGB_SIZE


def vandermonde(dim):
    """
    Creates a Vandermonde matrix of a given dimension, which can be used in fourier transforms
    :param dim: length and height of matrix
    :return: vandermonde matrix: array of shape (N,N) of complex128
    """
    indexes = np.meshgrid(range(dim), range(dim))
    return indexes[0] * indexes[1] * (2j * np.pi / dim)


def DFT(signal):
    """
    Transform a 1D discrete signal to its fourier representation.
    :param signal: array of float32 with shape (N,1)
    :return: fourier signal: array of complex128 with shape (N,1)
    """
    root = vandermonde(signal.shape[0])
    dft_matrix = np.exp(-1 * root).astype(np.complex128)
    fourier_signal = dft_matrix.dot(signal)
    return fourier_signal.astype(np.complex128)


def IDFT(fourier_signal):
    """
    Transform a 1D fourier representation back to the discrete signal.
    :param fourier_signal: array of complex128 with shape (N,1)
    :return: fourier signal: array of complex128 with shape (N,1)
    """
    root = vandermonde(fourier_signal.shape[0])
    idft_matrix = np.exp(root).astype(np.complex128)
    return idft_matrix.dot(fourier_signal) / fourier_signal.size


def DFT2(image):
    """
    Transform a 2D discrete signal to its fourier representation.
    :param image: grayscale image (matrix of pixels) of float32
    :return: fourier_image: 2D array of complex128
    """
    return DFT(DFT(image).T).T


def IDFT2(fourier_image):
    """
    Transform a 2D fourier representation back to the discrete signal.
    :param fourier_image: grayscale image (matrix of pixels) of float32
    :return: image: 2D array of complex128
    """
    return IDFT(IDFT(fourier_image).T).T


def conv_der(im):
    """
    Computes the magnitude of a given image's derivatives.
    :param im: grayscale image of float32
    :return: magnitude: grayscale image of float32
    """
    row_deriv = scipy.signal.convolve2d(im, DERIVE_VEC, mode='same', boundary='symm')
    col_deriv = scipy.signal.convolve2d(im, np.array(np.transpose(DERIVE_VEC)), mode='same', boundary='symm')
    return np.sqrt(np.square(col_deriv) + np.square(row_deriv))


def fourier_der(im):
    """
    Compute magnitude of image derivatives using fourier transform.
    :param im: float32 grayscale image
    :return: float32 grayscale image
    """
    fourier_im = DFT2(im)
    shifted = np.fft.fftshift(fourier_im)

    height = shifted.shape[0]
    width = shifted.shape[1]
    indexes = np.indices((height, width))
    row_indexes, col_indexes = indexes

    derived_x = shifted[:] * row_indexes * 2j * np.pi / (height * width)
    derived_y = shifted[:] * col_indexes * 2j * np.pi / (height * width)

    derived_shift_x = np.fft.ifftshift(derived_x)
    derived_shift_y = np.fft.ifftshift(derived_y)

    inverse_derv_x = IDFT2(derived_shift_x)
    inverse_derv_y = IDFT2(derived_shift_y)
    return np.sqrt((inverse_derv_x ** 2) + (inverse_derv_y ** 2)).real.astype(np.float32)


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


def blur_fourier(im, kernel_size):
    """
    Performs image blurring using 2d convolution between a given image and a guassian kernel
    :param im: image to be blurred, grayscale float32
    :param kernel_size: size of guassian kernel (odd integer)
    :return: blurry image, grayscale float32
    """
    kernel = get_Gauss_Kernel(kernel_size)
    padding = np.zeros(im.shape)
    w_start = (np.floor(padding.shape[0] / 2) - np.floor(kernel.shape[0] / 2)).astype(np.uint8)
    h_start = (np.floor(padding.shape[1] / 2) - np.floor(kernel.shape[1] / 2)).astype(np.uint8)

    padding[w_start: w_start + kernel.shape[0], h_start: h_start + kernel.shape[1]] = kernel
    padding = np.fft.ifftshift(padding)

    kernel_fourier = DFT2(padding)
    image_fourier = DFT2(im)
    deriv_fourier = np.multiply(kernel_fourier, image_fourier)
    return IDFT2(deriv_fourier).real.astype(np.float32)




# fourier
# nofars = 4.95831e-06
# mine = 2.70039e-11


# derv = conv_der(read_image("monkey.jpg", GRAY_SCALE))
# print(derv[50,50])

#conv
#nofar = 0.00881268357383
#mine = 0.00440634152427


