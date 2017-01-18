##############################################################################
# FILE: sol4.py
# WRITER: Keren Meron, keren.meron, 200039626
# EXERCISE: Image Processing ex4 2016-2017
# DESCRIPTION: creating a panorama from several images
##############################################################################

import sol4_utils as utils
import scipy
import numpy as np
import sol4_add
import matplotlib.pyplot as plt
from numpy import random


DERIVE_VEC = np.array([[1], [0], [-1]])
KERNEL_SIZE = 3
K = 0.04
GRAY_SCALE = 1
SPREAD_RADIUS = 3
SUB_IMG_WIDTH = 7
SUB_IMG_HEIGHT = 7
DESCRIPTOR_RADIUS = 3
LEVEL3_COORDS_CONV = 4
MIN_SCORE = 0.1
INLIER_TOL = 9
RANSAC_ITERS = 10000
BLEND_FACTOR = 120


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

    for i in np.arange(pos.shape[0]):
        temp = scipy.ndimage.map_coordinates(im, coords[:,i*(k*k):(i+1)*k*k], order=1, prefilter=False)
        descriptors[:,:,i] = temp.reshape((k,k))
        mean_dist = descriptors[:,:,i] - (descriptors[:,:,i]).mean()
        normed = np.linalg.norm(mean_dist)
        if normed != 0:
            descriptors[:,:,i] = mean_dist / normed
        else:
            descriptors[:,:,i] = np.zeros(descriptors[:,:,i].shape)

    return descriptors.astype(np.float32)


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

    #get 2 maximal values for columns and rows
    sorted_cols = np.argsort(S_match_weight, axis=0)[S_match_weight.shape[0]-2:,:]
    sorted_rows = np.argsort(S_match_weight, axis=1)[:, S_match_weight.shape[1]-2:]

    #set in S_match_weight 1 if 2max, otherwise 0
    S_max_cols = np.zeros(S_match_weight.shape)
    S_max_rows = np.zeros(S_match_weight.shape)

    for i in np.arange(sorted_cols.shape[1]):
        S_max_cols[sorted_cols[0, i], i] = 1
        S_max_cols[sorted_cols[1, i], i] = 1
    for i in np.arange(sorted_rows.shape[0]):
        S_max_rows[i, sorted_rows[i, 0]] = 1
        S_max_rows[i, sorted_rows[i, 1]] = 1

    #filter elements > min score and maximal in both rows and columns
    S_min_score_thresh = (S_match_weight > min_score).astype(np.uint)

    max_both = np.multiply(S_max_rows, S_max_cols)
    max_both = np.multiply(max_both, S_min_score_thresh)
    matches_indexes = np.argwhere(max_both == 1)

    match_ind1 = matches_indexes[:, 0]
    match_ind2 = matches_indexes[:, 1]

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
    transformed = np.nan_to_num(transformed)
    if not np.all(transformed[2, :]):
        return transformed[:2, :].T
    new_points = transformed[:2, :] / (transformed[2, :])
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
    while True:
        try:
            max_inliers = ransac_homography_helper(pos1, pos2, inlier_tol)
        except ValueError:
            continue
        break
    max_inliers_size = max_inliers.size
    for _ in np.arange(num_iters):
        try:
            inliers = ransac_homography_helper(pos1, pos2, inlier_tol)
            if inliers.size > max_inliers_size:
                max_inliers, max_inliers_size = inliers, inliers.size
        except ValueError:
            continue

    points1, points2 = (pos1[max_inliers]), (pos2[max_inliers])
    H12 = sol4_add.least_squares_homography(points1, points2)

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
    H12 = sol4_add.least_squares_homography(random_pos1, random_pos2)
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
    pos2[:, 0] = pos2[:, 0] + im1.shape[1]
    inlier_points1, inlier_points2 = pos1[inliers], pos2[inliers]

    outlier_indices = np.arange(pos1.shape[0])
    outlier_indices = np.delete(outlier_indices, inliers)
    outlier_points1, outlier_points2 = pos1[outlier_indices], pos2[outlier_indices]

    plt.figure
    plt.imshow(combined_img, plt.cm.gray)
    plt.plot((outlier_points1[:,0], outlier_points2[:,0]),(outlier_points1[:,1], outlier_points2[:,1]), mfc='r',
             c='b', lw=0.5, marker='.')
    plt.plot((inlier_points1[:,0], inlier_points2[:,0]),(inlier_points1[:,1], inlier_points2[:,1]), mfc='r',
             c='y', lw=0.5, marker='.')
    plt.show()


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
    #transformed points: [top left, top right, bottom left, bottom right, center], shape (N, 4, 2)
    transformed_points = transform_corners(ims, Hs)
    num_images = len(ims)

    #panorama boundaries
    ROW_AXIS, COL_AXIS = 0, 1
    x_min = np.floor(np.min(transformed_points[0, :, COL_AXIS]))
    x_max = np.ceil(np.max(transformed_points[num_images-1, :, COL_AXIS]))
    y_min = np.floor(np.min(transformed_points[:, :, ROW_AXIS]))
    y_max = np.ceil(np.max(transformed_points[:, :, ROW_AXIS]))
    p_width = np.abs(x_max - x_min).astype(np.int32)
    p_height = np.abs(y_max - y_min).astype(np.int32)
    panorama1 = np.zeros((p_height, p_width))

    pan_x_bounds = get_centers_helper(ims, Hs, x_min, x_max)
    canvas_bounds = pan_x_bounds + np.abs(x_min)

    #coordinates in panorama system
    x_range = np.linspace(x_min, x_max, p_width).astype(np.int32)
    y_range = np.linspace(y_min, y_max, p_height).astype(np.int32)
    x, y = np.meshgrid(x_range, y_range)

    return backwarping(panorama1, ims, Hs, p_height, p_width, canvas_bounds, x, y)


def backwarping(panorama, ims, Hs, p_height, p_width, canvas_bounds, x_mesh, y_mesh):
    """ Places images into panorama, after homographic transformations and blending. """

    num_images = len(ims)
    for k in np.arange(num_images):
        panorama2 = np.zeros((p_height, p_width))

        curr_x_min = (np.floor(canvas_bounds[k]) - BLEND_FACTOR).astype(np.int32)
        curr_x_max = (np.ceil(canvas_bounds[k + 1]) + BLEND_FACTOR).astype(np.int32)

        if k == 0:
            curr_x_min = 0

        curr_x_range = (x_mesh[:, curr_x_min: curr_x_max + 1].flatten())
        curr_y_range = (y_mesh[:, curr_x_min: curr_x_max + 1].flatten())

        pan_area = np.zeros(curr_x_range.size * 2).reshape(curr_x_range.size, 2)
        pan_area[:, 0] = curr_x_range
        pan_area[:, 1] = curr_y_range
        original_area = apply_homography(pan_area, np.linalg.inv(Hs[k]))

        to_interpolate = np.transpose(np.fliplr(original_area))
        intensities = scipy.ndimage.map_coordinates(ims[k], to_interpolate, order=1, prefilter=False)
        intensities = intensities.reshape(p_height, (intensities.size / p_height).astype(np.int32))
        if k == 0:
            panorama[:, :curr_x_max + 1] = intensities
        elif k == num_images - 1:
            panorama2[:, curr_x_min:] = intensities
        else:
            panorama2[:, curr_x_min:curr_x_max + 1] = intensities

        # blend together
        if k > 0:
            middle = curr_x_min + BLEND_FACTOR
            panorama = blend_panorama(panorama, panorama2, middle)

    return panorama


def transform_corners(ims, Hs):
    """
    Transform 4 corner coordinates in each image in ims by its corresponding homography matrix
    :param ims: list of grayscale images
    :param Hs: list of M 3x3 homography matrices transforming points from coordinate system i to ponorama's coordinates
    :return: array of shape (N, 4, 2) with transformed [x,y] points
    """
    num_images = len(ims)
    transformed = np.zeros(4 * 2 * num_images).reshape(num_images, 4, 2)

    for i in np.arange(num_images):
        im = ims[i]
        #order: TOP LEFT, BOTTOME LEFT, TOP RIGHT, BOTTOM RIGHT
        curr_corners = np.hstack(([0, 0], [im.shape[0]-1, 0], [0, im.shape[1]-1], [im.shape[0]-1, im.shape[1]-1]))
        curr_points = np.fliplr(curr_corners.reshape(4, 2))
        transformed[i] = np.fliplr(apply_homography(curr_points, (Hs[i])))
    return np.floor(transformed)


def get_centers_helper(ims, Hs, x_min, x_max):
    """
    Find centers between given images, in their homographic transformation representation
    :param ims: list of N grayscale images
    :param Hs: list of M 3x3 homography matrices transforming points from coordinate system i to ponorama's coordinates
    :param x_min, x_max: edge points of total panoramic image
    :return: list of N+1 points (including the minimal and maximal edges)
    """
    centers = [x_min]
    for i in np.arange(len(ims)-1):
            original_center1 = np.fliplr(np.array([[int(ims[i].shape[0]//2), int(ims[i].shape[1]//2)]]))
            original_center2 = np.fliplr(np.array([[int(ims[i+1].shape[0]//2), int(ims[i+1].shape[1]//2)]]))
            new_center_1 = (apply_homography(original_center1, (Hs[i])))
            new_center_2 = (apply_homography(original_center2, (Hs[i+1])))
            centers.append(int((new_center_1[:, 0] + new_center_2[:, 0])//2))
    centers.append(x_max)
    return centers


def blend_panorama(pan1, pan2, middle):
    """
    Blend two given images in a axis
    :param pan1, pan2: grayscale images
    :param middle: index of col to blend at
    :return: blended image
    """
    mask = np.ones(pan1.shape)
    mask[:, :middle + 1] = 0
    return utils.pyramid_blending(pan2, pan1, mask, max_levels=7, filter_size_im=11, filter_size_mask=11)


