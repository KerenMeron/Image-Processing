##############################################################################
# FILE: sol1.py
# WRITER: Keren Meron, keren.meron, 200039626
# EXERCISE: Image Processing ex1 2016-2017
# DESCRIPTION: Opening image from file, histogram equalization and quantization
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from skimage.color import rgb2gray

#######   CONSTANT VALUES FOR IMAGE PROCESSING EX1    #######
GRAY_SCALE = 1
RGB = 2
RGB_SIZE = 3
GRAY_COLOR = "Greys_r"
NORMALIZE_FACTOR = 255
RGB2YIQ_CONVERSION_MAT = [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]
DEFAULT = 333
BINS = 256
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


def imdisplay(filename, representation=DEFAULT):
    """
    Displays an image from a given filename in a given representation.
    :param filename: name of file containing image
    :param representation: e.g. 1 for grayscale, 2 for RGB, 333 for default (no change)
    """
    if representation == GRAY_SCALE:
        plt.imshow(read_image(filename, representation), cmap=GRAY_COLOR)
    else:
        plt.imshow(read_image(filename, representation))
    plt.show()


def isRGB(img):
    """
    Checks whether given image is in color RGB or grayscale
    :param img: given image, matrix of pixels
    :return: True if RGB, False if grayscale
    """
    return len(img.shape) == RGB_SIZE and img.shape[-1] == RGB_SIZE


def rgb2yiq(imRGB):
    """
    Transforms an RGB image into the YIQ color space.
    :param imRGB: float32 matrix of pixels(arrays), normalized to [0,1] with RGB values
    :return: float32 matrix, normalized to [0,1] with YIQ values
    """
    conversion_matrix = np.array(RGB2YIQ_CONVERSION_MAT).transpose()
    return np.dot(np.array(imRGB), conversion_matrix)


def yiq2rgb(imYIQ):
    """
    Transforms an YIQ image into the RGB color space.
    :param imYIQ: float32 matrix of pixels(arrays), normalized to [0,1] with YIQ values
    :return: float32 matrix, normalized to [0,1] with RGB values
    """
    conversion_matrix = np.linalg.inv(np.array(RGB2YIQ_CONVERSION_MAT)).transpose()
    return np.dot(np.array(imYIQ), conversion_matrix)


def cdf_equalizer(cdf):
    """
    Stretch cumulative histogram to reach 0 and 255 values.
    :param cdf: cumulative histogram, with values in [0,255]
    :return: stretched cumulative histogram, as array
    """
    numerator = cdf - cdf[np.flatnonzero(cdf)[0]]
    denominator = cdf[-1] - cdf[np.flatnonzero(cdf)[0]]
    return np.round(numerator / denominator * (BINS - 1))


def get_histogram_normed(image):
    """
    Compute the histogram and cumulative histogram of a given image.
    :param image: intensity Y array for an image
    :return: cumulative histogram (array) and histogram (array)
    """
    hist, bounds = np.histogram(image, np.arange(BINS + 1), range=(0, BINS))
    cdf = np.cumsum(hist) / image.size * (BINS - 1)
    if cdf[0] != 0 or cdf[-1] != (BINS - 1):
        cdf = cdf_equalizer(cdf)
    return np.round(cdf), hist


def get_Y_norm(image):
    """
    Extract the intensities from a given grayscale or RGB picture
    :param image: grayscale or RGB float32 image with values in [0,1]
    :return: tuple: (intensities, yiq)
                intesities: 1d vector with values in [0,255]
                yiq: float32 matrix, normalized to [0,1] with YIQ values
    """
    im_orig_Y = image
    im_orig_yiq = image
    if isRGB(image):
        im_orig_yiq = rgb2yiq(image)
        im_orig_Y = im_orig_yiq[:, :, 0]
    im_orig_Y_norm = im_orig_Y * (BINS - 1)
    return im_orig_Y_norm, im_orig_yiq


def set_Y_get_image(yiq_image, y_values, is_RGB, clip=False):
    """
    Update Y values of yiq image representation.
    :param yiq_image: original yiq values
    :param y_values: new Y values
    :param is_RGB: True if image is in RGB, False if grayscale
    :param clip: boolean, whether to clip final image values or not
    :return: image with updated y values. if image is not grayscale, returns image in RGB
    """
    im_new = y_values.copy()    #if grey
    if is_RGB:
        yiq_image[:, :, 0] = y_values
        im_new = yiq2rgb(yiq_image)
        if clip:
            im_new = np.clip(im_new, 0, 1)
    return im_new


def histogram_equalize(im_orig):
    """
    Performs histogram equalization of a given image.
    :param im_orig: grayscale or RGB float32 image with values in [0,1]
    :return: [im_eq, hist_orig, hist_eq] where:
                im_eq: equalized image grayscale or RGB float32 image with values in [0,1]
                hist_orig: 256 bin histogram of the original image
                hist_eq: 256 bin histogram of the equalized image
    """
    im_orig_Y_norm, im_orig_yiq = get_Y_norm(im_orig)
    cdf, hist = get_histogram_normed(im_orig_Y_norm)

    im_new_Y = cdf[im_orig_Y_norm.astype(np.uint8)]
    cdf_equalized, hist_equalized = get_histogram_normed(im_new_Y)
    im_new_Y /= (BINS - 1)

    im_new = set_Y_get_image(im_orig_yiq, im_new_Y, isRGB(im_orig), clip=True)
    return im_new, hist, hist_equalized


def get_initial_separators(cdf, shape, amount):
    """
    Get initial separators for the cdf
    :param cdf: cumulative histogram of an image
    :param shape: size of each segment
    :param amount: number of separators needed
    :return: array of with amount indexes
    """
    separators = np.zeros(amount)
    for i in range(amount - 1):
        sep_option = i * shape
        cdf_bigger = np.where(cdf >= sep_option)[0]
        separators[i] = cdf_bigger[0]
    separators[-1] = BINS
    return np.round(separators)


def quantum_per_segment(separators, amount, histogram):
    """
    Find average values (quantam values) per each segment, bounded by the separators
    :param separators: array of values with bound segments
    :param amount: number of quantam values needed
    :return: array of amount values, in [0,255]
    """
    new_quant = np.zeros(amount)
    weights = np.arange(BINS)
    for i in range(new_quant.size):
        curr_weights = weights[separators[i].astype(np.int):separators[i + 1].astype(np.int)]
        curr_hist = histogram[separators[i].astype(np.int):separators[i + 1].astype(np.int)]
        new_quant[i] = np.average(curr_weights, weights=curr_hist)
    return new_quant


def quantize_pixels(separators, quantifiers, image):
    """
    Quantize image pixels using a look up table created from quantifiers
    :param separators: bounds for the segments of original image's cdf
    :param quantifiers: array with values to change all pixels in each segment to
    :param image: 1d array with original image's intensities, in [0,255]
    :return: quantized image: 1d array with new intensities, in [0,1]
    """
    look_up_table = np.zeros(BINS)
    for k in range(quantifiers.size):
        look_up_table[separators[k].astype(np.int):separators[k + 1].astype(np.int)].fill(quantifiers[k])
    quantized_pic = look_up_table[image.astype(np.uint8)].astype(np.float32)
    return quantized_pic / (BINS - 1)


def update_separators_quantifiers(separators, quantifiers, histogram):
    """
    Find new separators as the average of two consecutive quantifiers
    Find new quantifiers as the average of two NEW consecutive separators
    :param separators: original (previous) array of separators
    :param quantifiers: original (previous) array of quantifiers
    :return: tuple (array of new separators, array of new quantifiers)
    """
    new_sep = np.zeros(separators.size)
    new_sep[0], new_sep[-1] = 0, BINS

    for i in range(quantifiers.size - 1):
        new_sep[i + 1] = (quantifiers[i] + quantifiers[i + 1]) / 2
    new_quant = quantum_per_segment(new_sep, quantifiers.size, histogram)
    return np.round(new_sep), new_quant


def quantize_error(histogram, separators, quantifiers):
    """
    Calculate the quantization error for an image quantization.
    :param histogram: histogram of image
    :param quantifiers: array of quantam values (intensities)
    :param separators: array of separators (intensities) bounding each quantam value
    :return: error of image, sum for all pixels
    """
    pixels = np.arange(BINS)
    error_sum = 0
    for seg in range(separators.size - 1):
        vec = pixels[separators[seg].astype(np.uint) : separators[seg+1].astype(np.uint)]
        error = (quantifiers[seg] - vec) ** 2 * histogram[vec]
        error_sum += np.sum(error)
    return error_sum


def get_minimal_separators(hist, separators, quantam_values, n_iter):
    """
    Find new separators for quantization, by taking the average of the previously found quantum values.
    Do this a given n_iter times or until result converges.
    Find new quantum values each iteration, as the weighted average of each segment.
    :param hist: histogram of image, array
    :param separators: array of previously found separators
    :param quantam_values: array of previously found quantam values
    :param n_iter: times to perform iterations
    :return: new separators, new quantum values, array of error per iteration
    """
    errors = np.array([])
    errors = np.append(errors, quantize_error(hist, separators, quantam_values))

    for i in range(1, n_iter):
        new_sep, new_quant = update_separators_quantifiers(separators, quantam_values, hist)
        errors = np.append(errors, quantize_error(hist, new_sep, new_quant))
        if np.array_equal(new_sep, separators):
            break
        separators, quantam_values = new_sep, new_quant
    return separators, quantam_values, errors


def initial_quantization(hist, cdf, pixel_intensities, n_quant):
    """
    Calculate separators and by which calcultate quantum values for a given image histogram
    :param hist: histogram of image, array
    :param cdf: cumulative histogram of image, array
    :param pixel_intensities: image intensities Y, array
    :param n_quant: number of quantum values required
    :return: separators, quantum values
    """
    norm_factor = pixel_intensities.size / n_quant
    separators = get_initial_separators(cdf, norm_factor, n_quant + 1)
    quantam_values = quantum_per_segment(separators, n_quant, hist)
    return separators, quantam_values


def quantize(im_orig, n_quant, n_iter):
    """
    Perform optimal quantization of a given grayscale or RGB image.
    :param im_orig: grayscale or RGB float32 with values in [0,1], image to be quantized
    :param n_quant: number of intensities the output image will have
    :param n_iter: maximum number of iterations of the optimization procedure
    :return: [im_quant, error] :
                im_quant - quantized output image
                error - array with shape (n_iter,) (or less) of the total intensities error for each iteration in the
                quantize procedure
    """
    #get cumulative histogram
    im_orig_Y_norm, im_orig_yiq = get_Y_norm(im_orig)
    hist, bounds = np.histogram(im_orig_Y_norm, np.arange(BINS + 1))
    cdf = np.cumsum(hist)

    #get separators & quantum values and quantize image pixels
    separators, quantam_values = initial_quantization(hist, cdf, im_orig_Y_norm, n_quant)
    separators, quantum_values, errors = get_minimal_separators(hist, separators, quantam_values, n_iter)
    quantized_pic = quantize_pixels(separators, quantum_values, im_orig_Y_norm)

    quantized_pic = set_Y_get_image(im_orig_yiq, quantized_pic, isRGB(im_orig))
    return [quantized_pic, errors]
