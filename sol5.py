##############################################################################
# FILE: sol5.py
# WRITER: Keren Meron, keren.meron, 200039626
# EXERCISE: Image Processing ex5 2016-2017
# DESCRIPTION:
##############################################################################

import sol5_utils
import random
from scipy.misc import imread
from scipy.ndimage import convolve
from skimage.color import rgb2gray
import numpy as np
from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, merge
from keras.optimizers import Adam


NORMALIZE_FACTOR = 255
DEFAULT = 333
FILE_ERROR_MSG = "Error in file."
GRAY_SCALE = 1
RGB_SIZE = 3
NUMBER_RES_BLOCKS = 5


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
    if representation == GRAY_SCALE and len(image.shape) == RGB_SIZE and image.shape[-1] >= RGB_SIZE:
        image = rgb2gray(image)
    elif len(image.shape) != RGB_SIZE:
        image = rgb2gray(image)     #if grey turn into 2d array
    elif image.shape[2] > RGB_SIZE:  #png
        image = image[:,:,:3]
    return image


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    Creates a generator which creates pairs of images and patches of their corruption.
    :param filenames: list of filenames of clean images
    :param batch_size: size of the batch of images
    :param corruption_func: function receiving an image (numpy array) and returns a randomly corrupted version
    :param crop_size: tuple (height,width) specifying the crop size of the patches
    :return: data_generator: generator object which outputs (source_batch, target_batch)
    """
    loaded_images = {}
    while batch_size > 0:
        image_index = random.randint(0, len(filenames))
        file_name = filenames[image_index]
        if file_name not in loaded_images:
            loaded_images[file_name] = read_image(file_name, GRAY_SCALE)
        source_batch = loaded_images[file_name]
        crop_rows = random.randint(0, source_batch.shape[0] - crop_size[0])
        crop_cols = random.randint(0, source_batch.shape[1] - crop_size[1])
        corrupted = corruption_func(source_batch)
        target_batch = corrupted[crop_rows:crop_rows+crop_size[0], crop_cols:crop_cols+crop_size[1]]
        source_batch -= 0.5
        target_batch -= 0.5
        batch_size -= 1
        yield source_batch, target_batch


def resblock(input_tensor, num_channels):
    """
    Creates a residual block represented as a tensor with convolutional layers
    :param input_tensor: tensor to perform layers on
    :param num_channels: for each convolution layer
    :return: output tensor of layers configured
    """
    res_tensor = Convolution2D(num_channels, 3, 3, mode='same')(input_tensor)
    res_tensor = Activation('relu')(res_tensor)
    res_tensor = Convolution2D(num_channels, 3, 3, mode='same')(res_tensor)
    return merge([input_tensor, res_tensor], mode='sum')


def build_nn_model(height, width, num_channels):
    """
    Builds a neural network model, with
    :param height, width: for input tensor creation
    :param num_channels: for all convolutional layers
    :return: model: represents input and output through a sequence residual blocks
    """
    input_tensor = Input(shape=(1, height, width))
    output_tensor = input_tensor
    for i in range(NUMBER_RES_BLOCKS):
        temp_tensor = output_tensor
        if i == NUMBER_RES_BLOCKS - 1:
            num_channels = 1
        output_tensor = resblock(temp_tensor, num_channels)
    return Model(input_tensor, output_tensor)


def train_model(model, images, corruption_func, batch_size, samples_per_epoch, num_epochs, num_valid_samples):
    """
    Divides the images into a training set and validation set and generates from each a dataset with a given batch
    size and corruption function. Compiles a given model, and trains it.
    :param model: general neural network model for image restoration
    :param images: list of file paths pointing to image files
    :param corruption_func: function receiving an image (numpy array) and returns a randomly corrupted version
    :param batch_size: size of batch examples for each iteration of SGD (stochastic gradient descent)
    :param samples_per_epoch: number of actual samples in each epoch
    :param num_epochs: number of epochs for which the optimization will run
    :param num_valid_samples: number of samples in validation set to test on aftery every epoch
    :return: num
    """
    images_slice = int(len(images) * 0.8)
    training_set, validation_set = images[:images_slice], images[images_slice:]
    trainset_gen = load_dataset(training_set, batch_size, corruption_func, model.input_shape)
    validset_gen = load_dataset(validation_set, batch_size, corruption_func, model.input_shape)
    model.compile(loss='mean_square_loss', optimizer=Adam(beta_2=0.9))
    model.fit_generator(trainset_gen, samples_per_epoch=samples_per_epoch, nb_epoch=num_epochs,
                        validation_data=validset_gen, nb_val_samples=num_valid_samples)


def restore_image(corrupted_image, base_model, num_channels):
    """
    Expands a given base model in order to restore a correct image from a given corrupted image.
    :param corrupted_image: grayscale image of shape (height, width) with values [0,1] float32
    :param base_model: a neural network trained to restore small patches, with inputs in [-0.5,0.5]
    :param num_channels: number of channels used in the base_model
    :return: restored image, with shape of corrupted_image
    """
    nn_model = build_nn_model(corrupted_image.shape[0], corrupted_image.shape[1], num_channels)
    nn_model.set_weights(base_model.get_weights())
    restored_image = nn_model.predict()[0]
    return np.clip(restored_image, 0, 1)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Random noise function applied to a given image.  A gaussian with random variance.
    :param image: grayscale image in [0,1] range of float32
    :param min_sigma, max_sigma: non-negative scalars for variance range
    :return: corrupted image, of same type as the input image
    """
    sigma = random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(0, sigma, image.shape) #todo mean=0 or middle of image?
    corrupted = image + noise
    return np.clip(corrupted, 0, 1)


def learn_denoising_model(quick_mode=False):
    """
    Creates and trains a denoising neural network model;
    :param quick_mode: for smaller sizes, quicker training
    :return: trained model, num_channels
    """
    images = sol5_utils.images_for_denoising()
    corrupt_func = lambda img: add_gaussian_noise(img, min_sigma=0, max_sigma=0.2)
    height, width, num_channels = 24, 24, 48
    model = build_nn_model(height, width, num_channels)

    batch_size, samples_per_epoch, num_epochs, num_valid_samples = 100, 10000, 5, 1000
    if quick_mode:
        batch_size, samples_per_epoch, num_epochs, num_valid_samples = 10, 30, 2, 30
    train_model(model, images, corrupt_func, batch_size, samples_per_epoch, num_epochs, num_valid_samples)

    return model, num_channels


def add_motion_blur(image, kernel_size, angle):
    """
    Adds a blur to a given image by convolving with a kernel at a certain angle
    :param image: grayscale image of [0,1] float32
    :param kernel_size: odd integer
    :param angle: angle in radians in [0,pi)
    :return: corrupted image, same type as input image
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return convolve(image, kernel) #todo mode?


def random_motion_blur(image, list_of_kernel_size):
    """
    Applies a motion blur with random kernel size and angle.
    :param image: grayscale image of [0,1] float32
    :param list_of_kernel_size: list of odd integers
    :return: corrupted image, same type as input image
    """
    angle = np.pi * random.random()
    index = random.randint(0, len(list_of_kernel_size))
    kernel_size = list_of_kernel_size[index]
    return add_motion_blur(image, kernel_size, angle)


def learn_deblurring_model(quick_mode=False):
    """
    Creates and trains a deblurring neural network model;
    :param quick_mode: for smaller sizes, quicker training
    :return: model, num_channels
    """
    images = sol5_utils.images_for_deblurring()
    corrupt_func = lambda img: random_motion_blur(img, [7]) #todo so why list?
    height, width, num_channels = 16, 16, 32
    model = build_nn_model(height, width, num_channels)

    batch_size, samples_per_epoch, num_epochs, num_valid_samples = 100, 10000, 10, 1000
    if quick_mode:
        batch_size, samples_per_epoch, num_epochs, num_valid_samples = 10, 30, 2, 30
    train_model(model, images, corrupt_func, batch_size, samples_per_epoch, num_epochs, num_valid_samples)

    return model, num_channels



if __name__ == "__main__":
    model, num_channels = learn_denoising_model(quick_mode=True)
