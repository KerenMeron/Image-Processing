##############################################################################
# FILE: sol3.py
# WRITER: Keren Meron, keren.meron, 200039626
# EXERCISE: Image Processing ex3 2016-2017
# DESCRIPTION: perform image blending using gaussian and laplacian pyramids
##############################################################################

import numpy as np
import scipy.signal
from scipy.misc import imread, imsave
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os

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


def blur_spatial_kernel(im, kernel):
    """
    Blur a given image by convolving with a given kernel
    :param im: grayscale image with double values in [0,1]
    :param kernel: 1D-row used for pyramid construction, normalized
    :return: blurred image
    """
    rows_blur = scipy.ndimage.filters.convolve(im, kernel, mode='nearest')
    return scipy.ndimage.filters.convolve(rows_blur, kernel.transpose(), mode='nearest')


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


def pad_zeros(im):
    """
    Pad image with zeros in uneven indices (between rows and columns, excluding boundaries)
    :param im: grayscale image with double values in [0,1]
    :return: padded im, same type as im
    """
    indices_row = np.arange(im.shape[0])
    indices_col = np.arange(im.shape[1])
    row_padded = np.insert(im, indices_row, 0, axis=0)
    return np.insert(row_padded, indices_col, 0, axis=1)


def expand(im, kernel):
    """
    Expands a given image to twice its size
    :param im: grayscale image with double values in [0,1]
    :return: expanded im, same type as im
    """
    padded_im = pad_zeros(im)
    return blur_spatial_kernel(padded_im, kernel * EXPAND_FACTOR)


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
        pyr.append(guassian_pyr[i] - expand(guassian_pyr[i + 1], filter_vec))
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
        img = expand(img, filter_vec) + lpyr[i]
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


def render_pyramid(pyr, levels):
    """
    Outlays images in a given pyramid on a black canvas, with images stretched in [0,1]
    :param pyr: laplacian or gaussian pyramid
    :param levels: number of levels in pyramid to present
    :return: canvas of images
    """
    width = int(canvas_size(levels, pyr[0].shape[1]))
    canvas = np.zeros(int(pyr[0].shape[0] * width)).reshape((int(pyr[0].shape[0])), width)
    col = 0
    for i in range(levels):
        stretched_im = (pyr[i] - np.min(pyr[i])) / (np.max(pyr[i]) - np.min(pyr[i]))
        canvas[:pyr[i].shape[0], col: col + pyr[i].shape[1]] = stretched_im
        col += pyr[i].shape[1]
    return canvas


def display_pyramid(pyr, levels):
    """
    Displays images from a given pyramid on a black canvas, with images stretched in [0,1]
    :param pyr: laplacian or gaussian pyramid
    :param levels: number of levels in pyramid to present
    """
    plt.imshow(render_pyramid(pyr, levels), plt.cm.gray)
    plt.show()


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
    max_levels = fix_levels(max_levels, im1.shape)
    laplac_im1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    laplac_im2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    mask_pyr = build_gaussian_pyramid(mask.astype(np.float32), max_levels, filter_size_mask)[0]
    laplacian_out = list()
    for i in range(max_levels):
        laplacian_out.append(np.multiply(mask_pyr[i], laplac_im1[i]) + np.multiply(1 - mask_pyr[i], laplac_im2[i]))
    coeff = np.ones(max_levels)
    im_blend = laplacian_to_image(laplacian_out, filter_vec, coeff).astype(np.float32)
    return np.clip(im_blend, 0, 1)



def blending_example1():
    max_levels = 3
    filter_size_im = 5
    filter_size_mask = 5
    im1 = read_image(relpath('externals/room.jpg'))
    im2 = read_image(relpath('externals/planet.jpg'))
    mask = read_image(relpath('externals/mask_window.png'), GRAY_SCALE).astype(np.bool)
    im_blend = blending_examples_helper(im1, im2, mask, max_levels, filter_size_im, filter_size_mask)
    return im1, im2, mask, im_blend


def blending_example2():
    max_levels = 3
    filter_size_im = 29
    filter_size_mask = 3
    im1 = read_image(relpath('externals/soup.jpg'))
    im2 = read_image(relpath('externals/woman.jpg'))
    mask = read_image(relpath('externals/mask_soup.png'), GRAY_SCALE).astype(np.bool)
    im_blend = blending_examples_helper(im1, im2, mask, max_levels, filter_size_im, filter_size_mask)
    return im1, im2, mask, im_blend


def blending_examples_helper(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):

    blended_R = pyramid_blending(im1[:,:,0], im2[:,:,0], mask, max_levels, filter_size_im, filter_size_mask)
    blended_G = pyramid_blending(im1[:,:,1], im2[:,:,1], mask, max_levels, filter_size_im, filter_size_mask)
    blended_B = pyramid_blending(im1[:,:,2], im2[:,:,2], mask, max_levels, filter_size_im, filter_size_mask)
    blended = np.zeros(im1.size).reshape(im1.shape)
    blended[:,:,0] = blended_R
    blended[:,:,1] = blended_G
    blended[:,:,2] = blended_B

    fig = plt.figure()
    fig.add_subplot(221)
    plt.imshow(im1)
    fig.add_subplot(222)
    plt.imshow(im2)
    fig.add_subplot(223)
    plt.imshow(mask, plt.cm.gray)
    fig.add_subplot(224)
    plt.imshow(blended)
    plt.show()
    return blended.astype(np.float32)


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

