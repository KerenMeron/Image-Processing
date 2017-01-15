# Naomi Deutsch
# 203516083
# naomid

import sol4 as keren

import os
import sol4_utils_naomi as myFunc
import sol4_add as schoolFunc
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
import scipy.ndimage as map
from scipy.signal import convolve as conv
from skimage.color import rgb2gray
from scipy.misc import imread, imshow

CONV_MATRIX = np.array([1, 0, -1]).reshape(3, 1).astype(np.float32)
K = 0.04


def harris_corner_detector(im):
    """
    Harris corner detector function- detect points in a
    frame that can be localized well and that will be likely reproduced in the consecutive frame.
    :param im: grayscale image to find key points inside.
    :return: An array with shape (N,2) of [x,y] key points locations in im
    """

    # calculating dx
    I_x = conv(im, CONV_MATRIX, mode='same')
    # calculating dy
    I_y = conv(im, CONV_MATRIX.transpose(), mode='same')

    blur_I_x2= myFunc.blur_spatial(I_x**2, 3)
    blur_I_y2 = myFunc.blur_spatial(I_y**2, 3)
    blur_I_x_I_Y = myFunc.blur_spatial(I_x*I_y, 3)

    # response image
    R = (blur_I_x2 * blur_I_y2 - blur_I_x_I_Y**2) - K * (blur_I_x2 + blur_I_y2)**2
    l_m_p = schoolFunc.non_maximum_suppression(R)
    x, y = np.where(l_m_p == True)
    pos = np.array(list(zip(y, x)))

    return pos


def sample_descriptor(im, pos, desc_rad):
    """
    Descriptor sampling
    :param im: grayscale image to sample within.
    :param pos: An array with shape (N,2) of [x,y] positions to sample descriptors in im.
    :param desc_rad: ”Radius” of descriptors to compute
    :return: desc:  A 3D array with shape (K,K,N) containing the i-th descriptor at desc(:,:,i).
            The per−descriptor dimensions KxK are related to the desc rad argument as follows K = 1+2∗desc rad.
    """

    # here we calculate a matrix that we will subtract from pos[i] and we will get
    # a window of size (size*size) and it will hold all the coordinates (x,y) around pos[i]
    x = np.arange(-desc_rad, desc_rad + 1)
    size = 2*desc_rad+1
    new_x = np.repeat(x, size).reshape(size,size)
    coor_array = np.dstack((new_x , np.transpose(new_x)))

    # the descriptors array
    desc = np.zeros((size, size, pos.shape[0]))

    for i in range(pos.shape[0]):

        x_cor = np.tile((np.arange(pos[i][1] - desc_rad,pos[i][1] + desc_rad + 1).reshape(7,1)),7)
        y_cor = np.tile((np.arange(pos[i][0] - desc_rad,pos[i][0] + desc_rad + 1).reshape(7,1)),7).T

        #map_coor = map.map_coordinates(im, tmp.reshape(2,size**2), order=1, prefilter=False)
        map_coor = map.map_coordinates(im, [x_cor, y_cor], order=1, prefilter=False)
        temp = map_coor - np.mean(map_coor)
        norm = np.linalg.norm(temp)
        if norm == 0:
            continue
        else:
            d = (temp / norm)
            desc[:,:,i] = d

    return desc


def find_features(pyr):
    """
    This function is responsible for both the feature detection and the descriptor extraction.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: pos: An array with shape (N,2) of [x,y] feature location per row found in the image.
                  These coordinates are provided at the pyramid level pyr[0].
             desc: A feature descriptor array with shape (K,K,N).
    """
    pos = schoolFunc.spread_out_corners(pyr[0], 7, 7, 12)
    sample_pos = pos / 4
    desc = sample_descriptor(pyr[2], sample_pos, 3)
    return pos, desc


def match_features(desc1, desc2, min_score):
    """
    Thus function performing the matching procedure.
    :param desc1: A feature descriptor array with shape (K,K,N1).
    :param desc2: A feature descriptor array with shape (K,K,N2).
    :param min_score: Minimal match score between two descriptors required to be regarded as corresponding points.
    :return: match_ind1: Array with shape (M,) and dtype int of matching indices in desc1.
             match_ind2: Array with shape (M,) and dtype int of matching indices in desc2.
    """
    K1,_,N1 = desc1.shape
    desc1 = desc1.reshape(-1, desc1.shape[-1])
    desc1_f = desc1.T

    K2,_,N2 = desc2.shape
    desc2 = desc2.reshape(-1, desc2.shape[-1])
    desc2_f = desc2.T

    # desc1_f = desc1.reshape((desc1.shape[0]**2),desc1.shape[2]).T
    # desc2_f = desc2.reshape((desc2.shape[0]**2),desc2.shape[2])

    res_1 = np.dot(desc1_f, desc2_f.T)
    # res_2 = np.dot(desc2_f, np.transpose(desc1_f))

    res_1[np.where(res_1 < min_score)] = 0
    # res_2[np.where(res_2 < min_score)] = 0

    # max_1 = np.argsort(res_1, axis=1)[:,res_1.shape[1]-2:]
    # max_2 = np.argsort(res_2, axis=1)[:,res_2.shape[1]-2:]

    max_1 = np.argsort(res_1, axis=1)[:, res_1.shape[1] - 2:]
    max_2 = np.argsort(res_1.T, axis=1)[:,res_1.T.shape[1] - 2:]

    zeros_1 = np.zeros((N1,N2))
    zeros_2 = np.zeros((N2,N1))

    for i in range(N1):
        zeros_1[i, max_1[i,0]] = 1
        zeros_1[i, max_1[i,1]] = 1

    for i in range(N2):
        zeros_2[i, max_2[i,0]] = 1
        zeros_2[i, max_2[i,1]] = 1

    check = zeros_1 * zeros_2.T
    match_ind1, match_ind2 = np.where(check == 1)

    return match_ind1, match_ind2


def apply_homography(pos1, H12):
    #TODO check what to do if  pos_tilda[2:,] =0
    """
    A function that applies a homography transformation on a set of points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: pos2: An array with the same shape as pos1 with [x,y] point coordinates in image i+1 obtained from
                   transforming pos1 using H12.
    """
    new_pos = np.insert(pos1,2,1, axis=1)

    pos_tilda = np.dot(H12, new_pos.T)
    pos2 = pos_tilda[:2] / pos_tilda[2:,]
    # pos2 = pos_tilda / pos_tilda[2:,]

    return np.round(pos2).T


def ransac_homography(pos1, pos2, num_iters, inlier_tol):
    """
    RANSAC function
    :param pos1, pos2: Two Arrays, each with shape (N,2) containing n rows of [x,y] coordinates of matched points
    :param num_iters: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :return: H12: A 3x3 normalized homography matrix.
             inliers: An Array with shape (S,) where S is the number of inliers,
             containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    indx = np.arange(pos1.shape[0])
    inliers = np.array([])

    for i in range(num_iters):
        points = np.random.permutation(indx)[:4]
        H12 = schoolFunc.least_squares_homography(pos1[points], pos2[points])
        if H12 is not None:
            check_pos = apply_homography(pos1, H12)
            E = np.linalg.norm((check_pos-pos2), 2, axis=1)
            check_pos = np.argwhere(E < inlier_tol)
            if check_pos.size > inliers.size:
                inliers = check_pos

    inliers = inliers.flatten()

    H12 = schoolFunc.least_squares_homography(pos1[inliers], pos2[inliers])
    return H12, inliers


def display_matches(im1, im2, pos1, pos2, inliers):

    im = np.hstack((im1, im2))
    plt.imshow(im, cmap=plt.cm.gray)
    plt.scatter(pos1[:, 0], pos1[:, 1], marker='.', color = 'r')
    plt.scatter(im1.shape[1] + pos2[:, 0], pos2[:, 1], marker='.', color = 'r')

    outliers = np.arange(pos1.shape[0])
    bad_pos10 = np.delete(outliers, inliers)

    plt.plot((pos1[bad_pos10][:, 0], im1.shape[1] + pos2[bad_pos10][:, 0]),(pos1[bad_pos10][:, 1], pos2[bad_pos10][:, 1]), color='b', lw=0.5 )
    # plt.plot((pos1[inliers][:, 0], im1.shape[1] + pos2[inliers][:, 0]),(pos1[inliers][:, 1], pos2[inliers][:, 1]), color='y', lw=0.5 )

    plt.show()
    return


def accumulate_homographies(H_successive, m):
    """

    :param H_successive: A list of M−1 3x3 homography matrices where H_successive[i] is a homography that transforms
                         points from coordinate system i to coordinate system i+1
    :param m: Index of the coordinate system we would like to accumulate the given homographies towards.
    :return: H2m: A list of M 3x3 homography matrices, where H2m[i] transforms points from coordinate system i to coordinate
                   system m.
    """
    H2m = [0]*(len(H_successive)+1)
    H2m[m] = np.eye(3)
    H2m[m-1] = H_successive[m-1]
    for i in range(len(H_successive)):
        if i < m:
            pass



def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

if __name__ == "__main__":
    im1 = myFunc.read_image(relpath('external/office1.jpg'), 1)
    im2 = myFunc.read_image(relpath('external/office1.jpg'), 1)

    pyr1, vec1 = myFunc.build_gaussian_pyramid(im1, 3, 5)
    pyr2, vec2 = myFunc.build_gaussian_pyramid(im2, 3, 5)

    pos1, desc1 = find_features(pyr1)
    pos2, desc2 = find_features(pyr2)

    match_ind1, match_ind2 = match_features(desc1,desc2, 0.3)
    print(match_ind1.shape)
    H12, inliers = ransac_homography(pos1[match_ind1], pos2[match_ind2], 500, 20)
    print("namoi",H12)
    H12, inliers = keren.ransac_homography(pos1[match_ind1], pos2[match_ind2], 500, 20)
    print("keren",H12)
    # display_matches(im1, im2, pos1[match_ind1], pos2[match_ind2], inliers)

    # a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # index = [2, 3, 6]
    #
    # new_a = np.delete(a, index)
    # print(new_a)
