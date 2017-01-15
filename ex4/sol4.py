##############################################################################
# FILE: sol4.py
# WRITER: Keren Meron, keren.meron, 200039626
# EXERCISE: Image Processing ex4 2016-2017
# DESCRIPTION:
##############################################################################

import sol4_utils as utils
import scipy
import numpy as np
import sol4_add
import matplotlib.pyplot as plt
from numpy import random

import sol4_eldan #todo DELETE



DERIVE_VEC = np.array([[1], [0], [-1]])
KERNEL_SIZE = 3
K = 0.04
GRAY_SCALE = 1
SPREAD_RADIUS = 10
SUB_IMG_WIDTH = 7
SUB_IMG_HEIGHT = 7
DESCRIPTOR_RADIUS = 3
LEVEL3_COORDS_CONV = 4
MIN_SCORE = 0.1
INLIER_TOL = 20
RANSAC_ITERS = 500


def harris_corner_detector(im):
    """
    Finds feature points in a given image using harris corner detection algorithm
    :param im: grayscale image to find key points inside
    :return: pos - array of shape (N,2) of [x,y] key points' locations in im
    """

    Ix = scipy.signal.convolve2d(im, DERIVE_VEC, mode='same')
    Iy = scipy.signal.convolve2d(im, np.array(np.transpose(DERIVE_VEC)), mode='same')
    kernel = utils.get_Gauss_Kernel(KERNEL_SIZE)

    Ix_pow_blur = utils.blur_spatial_kernel(np.power(Ix, 2), kernel)
    Iy_pow_blur = utils.blur_spatial_kernel(np.power(Iy, 2), kernel)
    Ix_Iy_blur = utils.blur_spatial_kernel(np.multiply(Ix, Iy), kernel)

    response = response_from_derivatives(Ix_pow_blur, Ix_Iy_blur, Iy_pow_blur)
    max_responses = sol4_add.non_maximum_suppression(response)
    # coords = np.column_stack(np.where(max_responses))
    corners = np.nonzero(max_responses)
    return np.fliplr(np.array(list(np.transpose(corners))))


def response_from_derivatives(A, BC, D):
    """
    Finds the corner response office1of an image, to use for harris corner detector
    A, B, C, D construct a matrix M, for which we find response:
    R = det(M) - k(trace(M))**2
    :param A: x-derivatives, squared, blurred
    :param BC: x-derivatives * y-derivatives, blurred
    :param D: y-derivatives, squared, blurred
    :return: matrix R, corner response for each pixel
    """
    det = np.multiply(A, D) - np.power(BC, 2)
    trace = np.add(A, D)
    return det - K * np.power(trace, 2)


def sample_descriptor(im, pos, desc_rad):
    """
    Sample normalized kxk patch around each given position in a given image
    :param im: grayscale image to sample within, assumes from 3rd level in pyramid
    :param pos: array of shape (N,2) of (x,y) positions to sample descriptors in im
    :param desc_rad: radius of descriptors to compute
    :return: desc: 3D array of shape (K,K,N) containing the i'th descriptor at desc(:,:,i). K=1+2*desc_rand
    """
    k = 2 * desc_rad + 1
    range = np.linspace(-desc_rad, desc_rad, k)

    x, x_range = np.meshgrid(range, pos[:, 0])
    y, y_range = np.meshgrid(range, pos[:, 1])

    x_coords = np.tile(x + x_range, k)
    y_coords = (y + y_range).repeat(k)
    coords = np.row_stack((y_coords.flatten(), x_coords.flatten()))
    descriptors = np.zeros((k * k * len(pos))).reshape(k, k, len(pos))

    for i in np.arange(len(pos)):
        temp = scipy.ndimage.map_coordinates(im, coords[:,i*(k*k):(i+1)*k*k], order=1, prefilter=False)
        descriptors[:,:,i] = temp.reshape((k,k))
        mean_dist = descriptors[:,:,i] - (descriptors[:,:,i]).mean()
        normed = np.linalg.norm(mean_dist)
        if normed != 0:
            descriptors[:,:,i] = mean_dist / normed
        else:
            descriptors[:,:,i] = np.zeros(descriptors[:,:,i].shape)

    return descriptors


def find_features(pyr):
    """
    Wrapper function for finding feature points and creating descriptors for them
    :param pyr: gaussian pyramid of a grayscale image with 3 levels
    :return:
            pos: array of shape (N,2) of [x,y] feature locations found in image, from level 0 in pyramid
            desc: feature descriptor array of shape (K,K,N)
    """
    features = sol4_add.spread_out_corners(pyr[0], SUB_IMG_HEIGHT, SUB_IMG_WIDTH, SPREAD_RADIUS)
    descriptors = sample_descriptor(pyr[2], features / LEVEL3_COORDS_CONV, DESCRIPTOR_RADIUS)
    return features, descriptors


def match_features(desc1, desc2, min_score):
    """
    Matches descriptors of features in two images, finding only those which match each other
    :param desc1: feature descriptor array of shape (K,K,N1)
    :param desc2: feature descriptor array of shape (K,K,N2)
    :param min_score: minimal match score between two descriptors
    :return:
            match_ind1: array of shape (M,) and dtype int of matching indices in desc1
            match_ind2: array of shape (M,) and dtype int of matching indices in desc2
    """
    N1, N2, K = desc1.shape[2], desc2.shape[2], desc1.shape[0]
    desc1_mat = desc1.reshape(-1, N1).astype(np.float32).T
    desc2_mat = desc2.reshape(-1, N2).astype(np.float32)

    S_match_weight = np.dot(desc1_mat, desc2_mat).astype(np.float32)
    # S_match_weight = np.array([[7,9,3],[5,6,4],[2,8,1]])

    #get 2 maximal values for columns and rows
    sorted_cols = np.argsort(S_match_weight, axis=0)[S_match_weight.shape[0]-2:,:]
    sorted_rows = np.argsort(S_match_weight, axis=1)[:,S_match_weight.shape[1]-2:]

    #set in S_match_weight 1 if 2max, otherwise 0
    S_max_cols = np.zeros(S_match_weight.shape)
    S_max_rows = np.zeros(S_match_weight.shape)

    for i in range(sorted_cols.shape[1]):
        S_max_cols[sorted_cols[0, i], i] = 1
        S_max_cols[sorted_cols[1, i], i] = 1
    for i in range(sorted_rows.shape[0]):
        S_max_rows[i, sorted_rows[i, 0]] = 1
        S_max_rows[i, sorted_rows[i, 1]] = 1

    #filter elements > min score and maximal in both rows and columns
    S_min_score_thresh = (S_match_weight > min_score).astype(np.uint)

    max_both = np.multiply(S_max_rows, S_max_cols)
    max_both = np.multiply(max_both, S_min_score_thresh)
    matches_indexes = np.argwhere(max_both == 1)

    match_ind1 = matches_indexes[:,0]
    match_ind2 = matches_indexes[:,1]

    return match_ind1, match_ind2


def apply_homography(pos1, H12):
    """
    Applies homographic transformation on a given set of points
    :param pos1: array of shape (N,2) of[x,y] point coordinates
    :param H12: 3x3 homography matrix
    :return: pos2: array with shape of pos1 in another image, from transformation on pos1
    """
    prev_points = np.insert(pos1, 2, 1, axis=1)
    transformed = np.dot(H12, prev_points.T)
    new_points = transformed[:2, :] / (transformed[2, :])
    new_points = np.nan_to_num(new_points)
    return new_points.T


def ransac_homography(pos1, pos2, num_iters, inlier_tol):
    """
    Performs RANSAC iterations in order to find the inlier match set as points which match from two given points
    sets, and the matching homographic transformation between them.
    :param pos1, pos2: arrays, each of shape (N,2) of [x,y] coordinate points
    :param num_iters: number of RANSAC iterations to perform
    :param inlier_tol: inlier tolerance threshold
    :return:
            H12: 3x3 normalized homography matrix
            inliers: array of shape (S,) of indices in pos1/pos2 of the max set of inliers found
            if no inliers found, returns (None, None)
    """
    #todo fix all np.arange back to regular python range() in loops

    max_inliers = ransac_homography_helper(pos1, pos2, inlier_tol)
    max_inliers_size = max_inliers.size
    for _ in np.arange(num_iters):
        try:
            inliers = ransac_homography_helper(pos1, pos2, inlier_tol)
            if inliers.size > max_inliers_size:
                max_inliers, max_inliers_size = inliers, inliers.size
        except ValueError:
            continue

    points1, points2 = (pos1[max_inliers]), (pos2[max_inliers])
    H12 = sol4_add.least_squares_homography(points2, points1)

    if H12 is None: #todo remove
        print("H none")
    # transformed = apply_homography(pos1, H12)
    # error = np.linalg.norm((transformed - pos2), axis=1) ** 2
    # final_inlier_indices = np.where(error < inlier_tol)[0]
    # print("indice===========\n", final_inlier_indices.shape)
    # print("ransac:\n\n", H12, final_inlier_indices.reshape(final_inlier_indices.size,))
    return H12, max_inliers.reshape(max_inliers.size,)


def ransac_homography_helper(pos1, pos2, inlier_tol):
    """
    Performs a single RANSAC iterations, i.e. randomly selects 2 sets of 4 points, finds their homography matrix,
    applies this matrix on all points in pos1 and compute min square error from pos1. Mark as inliers only points
    with error smaller than a given threshold.
    :param pos1, pos2: arrays, each of shape (N,2) of [x,y] coordinate points
    :param inlier_tol: inlier tolerance threshold
    :return: inlier_indices: indices of found inliers from pos1/pos2
    :raise: ValueError if transform matrix was not found
    """
    rand_range = np.arange(pos1.shape[0])
    random_indices = random.permutation(rand_range)[:4]
    random_pos1 = pos1[random_indices]
    random_pos2 = pos2[random_indices]
    # print("=== NEW ITER POINTS ===", random_indices)
    H12 = sol4_add.least_squares_homography(random_pos1, random_pos2)
    # print("--- H ---", H12)
    if H12 is None:
        raise ValueError

    transformed = apply_homography(pos1, H12)
    error = np.linalg.norm((transformed - pos2), axis=1) ** 2
    inlier_indices = np.where(error < inlier_tol)[0]
    return inlier_indices.flatten()


def display_matches(im1, im2, pos1, pos2, inliers):
    """
    Displays matched points found between two given images
    :param im1, im2: grayscale images
    :param pos1, pos2: arrays of shape (N,2) containing matched point coordinates
    :param inliers: array of shape (S,) with inlier matches (indices for pos1/pos2)
    """
    combined_img = np.hstack((im1, im2))
    pos2[:,0] = pos2[:,0] + im1.shape[1]
    inlier_points1, inlier_points2 = pos1[inliers], pos2[inliers]

    outlier_indices = np.arange(pos1.shape[0])
    outlier_indices = np.delete(outlier_indices, inliers)
    outlier_points1, outlier_points2 = pos1[outlier_indices], pos2[outlier_indices]

    # plt.figure
    # plt.imshow(combined_img, plt.cm.gray)
    # plt.plot((inlier_points1[:,0], inlier_points2[:,0]),(inlier_points1[:,1], inlier_points2[:,1]), mfc='r',
    #          c='y', lw=0.5, marker='.')
    # plt.plot((outlier_points1[:,0], outlier_points2[:,0]),(outlier_points1[:,1], outlier_points2[:,1]), mfc='r',
    #          c='b', lw=0.5, marker='.')
    # plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Fits homographis to transform from some coordinate system to a given coordinate system. Transforms given
    homographic matrices so that they are relative to a given coordinate system of the matrix at index m.
    :param H_succesive: list of (M-1) 3x3 homography matrices transforming points from c. system i to i+!
    :param m: index of coordinate system we want to accumulate points towards
    :return: H2m: list of M 3x3 homography matrices transforming points from c. system i to m
    """
    M = len(H_succesive) + 1
    H2m = [0] * M
    H2m[m] = np.eye(3)
    for i in np.arange(m-1, -1, -1):
        H2m[i] = np.dot(H2m[i+1], H_succesive[i])
        H2m[i] = H2m[i] / H2m[i][2,2]
    for j in np.arange(m+1, M):
        H2m[j] = np.dot(H2m[j-1], np.linalg.inv(H_succesive[j-1]))
        H2m[j] = H2m[j] / H2m[j][2,2]
    return H2m


def render_panorama(ims, Hs):
    """
    Place and stitch several given images into one panorama image.
    Panorama image composed of vertical strips, backwarped using given homographies for each given image
    :param ims: list M of grayscale images
    :param Hs: list of M 3x3 homography matrices transforming points from coordinate system i to ponorama's coordinates
    :return: panorama: grayscale image composed of vertical strips
    """

    #transformed points: [top left, top right, bottom left, bottom right, center],   shape (N, 5, 2)
    transformed_points = transform_corners_center(ims, Hs)
    num_images = len(ims)
    print("transformed points,\n ", transformed_points)

    #panorama boundaries
    ROW_AXIS, COL_AXIS = 0, 1
    TOP_LEFT, BOTTOM_LEFT, TOP_RIGHT, BOTTOM_RIGHT, CENTER = 0, 1, 2, 3, 4
    x_min = np.floor(np.min(transformed_points[0, :, ROW_AXIS]))
    x_max = np.floor(np.max(transformed_points[num_images-1, :, ROW_AXIS]))
    y_min = np.floor(np.min(transformed_points[:, :, COL_AXIS]))
    y_max = np.ceil(np.max(transformed_points[:, :, COL_AXIS]))
    p_width = np.abs(x_max - x_min)
    p_height = np.abs(y_max - y_min)
    panorama = np.zeros(p_height * p_width).reshape(p_height, p_width)

    #vertical strips in panorama
    pan_x_bounds = np.zeros(num_images+1)
    pan_x_bounds[0] = x_min
    for j in np.arange(1, num_images):
        mid = (transformed_points[j-1, CENTER, ROW_AXIS] + transformed_points[j, CENTER, ROW_AXIS]) / 2
        pan_x_bounds[j] = transformed_points[j-1, CENTER, COL_AXIS] + mid.astype(np.int32)
    pan_x_bounds[num_images] = x_max
    canvas_bounds = pan_x_bounds + np.abs(x_min)
    print("panoram bounds: ", pan_x_bounds)

    #coordinates in panorama system
    x_range = np.linspace(x_min, x_max, p_width).astype(np.int32)
    y_range = np.linspace(y_min, y_max, p_height).astype(np.int32)
    x, y = np.meshgrid(x_range, y_range)

    #backwarping
    for k in np.arange(num_images):
        curr_x_min = np.floor(canvas_bounds[k])
        curr_x_max = np.ceil(canvas_bounds[k+1])
        # print("keren bounds",curr_x_min,curr_x_max+1)
        curr_x_range = (x[:, curr_x_min:curr_x_max+1].flatten())
        curr_y_range = (y[:, curr_x_min:curr_x_max+1].flatten())

        pan_area = np.zeros(curr_x_range.size * 2).reshape(curr_x_range.size, 2)
        pan_area[:, 0] = curr_x_range
        pan_area[:, 1] = curr_y_range
        original_area = apply_homography(pan_area, Hs[k])

        to_interpolate = np.transpose(np.fliplr(original_area))
        print("to interpolate", to_interpolate.shape)
        intensities = scipy.ndimage.map_coordinates(ims[k], to_interpolate, order=1, prefilter=False)
        if k == num_images - 1:
            panorama[:, canvas_bounds[k]:] = intensities.reshape(p_height, intensities.shape / p_height)
        else:
            panorama[:, canvas_bounds[k]:canvas_bounds[k+1]+1] = intensities.reshape(p_height, curr_x_max-curr_x_min+1)

    return panorama



def transform_corners_center(ims, Hs):
    """
    Transform 4 corner coordinates and center in each image in ims by its corresponding homography matrix
    :param ims: list of grayscale images
    :param Hs: list of M 3x3 homography matrices transforming points from coordinate system i to ponorama's coordinates
    :return: array of shape (N, 5, 2) with transformed [x,y] points
    """
    num_images = len(ims)
    transformed = np.zeros(5 * 2 * num_images).reshape(num_images, 5, 2)

    for i in np.arange(num_images):
        im = ims[i]
        #order: TOP LEFT, BOTTOME LEFT, TOP RIGHT, BOTTOM RIGHT, CENTER
        curr_corners = np.hstack(([0, 0], [im.shape[0]-1, 0], [0, im.shape[1]-1], [im.shape[0]-1, im.shape[1]-1]))
        center = [np.ceil(im.shape[0]/2), np.ceil(im.shape[1]/2)]
        curr_points = np.fliplr(np.hstack((curr_corners, center)).reshape(5,2))
        transformed[i] = apply_homography(curr_points, np.linalg.inv(Hs[i]))

    return np.floor(transformed)

def test():
    img1 = utils.read_image('external/office1.jpg', GRAY_SCALE)
    img2 = utils.read_image('external/office1.jpg', GRAY_SCALE)
    # img3 = utils.read_image('external/office3.jpg', GRAY_SCALE)
    # img4 = utils.read_image('external/office4.jpg', GRAY_SCALE)


    pyr1 = utils.build_gaussian_pyramid(img1, 3, 3)
    pyr2 = utils.build_gaussian_pyramid(img2, 3, 3)
    feat1, desc1 = find_features(np.array(pyr1[0]))
    feat2, desc2 = find_features(np.array(pyr2[0]))
    ind1, ind2 = match_features(desc1, desc2, MIN_SCORE)
    feat1, feat2 = feat1[ind1,:], feat2[ind2,:]

    H12, inliers12 = ransac_homography(feat1, feat2, RANSAC_ITERS, INLIER_TOL)
    display_matches(img1, img2, feat1, feat2, inliers12)


def test_ransac(im1, im2):
    pass
    # H12, inliers = sol4_naomi.ransac_homography(pos1, pos2, RANSAC_ITERS, INLIER_TOL)
    # print("==NAOMI==\nH12: \n", H12, "inliers:\n", inliers)
    # H12, inliers = ransac_homography(pos1, pos2, RANSAC_ITERS, INLIER_TOL)
    # print("==KEREN==\nH12: \n", H12, "inliers:\n", inliers)

def test_matching(desc1, desc2):
    match_features(desc1, desc2, MIN_SCORE)


def test_harris(img):
    features = sol4_add.spread_out_corners(img, SUB_IMG_HEIGHT, SUB_IMG_WIDTH, SPREAD_RADIUS)
    # features = sol4.harris_corner_detector(img)
    x_coords = features[:,0]
    y_coords = features[:,1]
    plt.imshow(img, plt.cm.gray)
    plt.plot(x_coords, y_coords, '.')
    plt.show()
    return features

def test_mops(img, pos):
    descriptors = sample_descriptor(img, pos, DESCRIPTOR_RADIUS)
    return descriptors

def test_homography():
    H = np.ones(9).reshape(3,3)
    pos = np.arange(4).reshape((2,2))
    apply_homography(pos, H)

if __name__ == "__main__":
    img1 = utils.read_image('external/office1.jpg', GRAY_SCALE)
    img2 = utils.read_image('external/office1.jpg', GRAY_SCALE)
    test_harris(img1)


