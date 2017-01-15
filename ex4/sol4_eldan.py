#############################################################
# FILE :sol4.py
# WRITER : Eldan chodorov , eldan , 201335965
# EXERCISE : image Proc sol4 2016-2017
#############################################################
import numpy as np
from scipy.misc import imread as imread
from scipy import ndimage
from scipy.ndimage import interpolation as ter
from scipy.signal import convolve2d
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from copy import *
from functools import reduce
import os
from PIL import ImageFile

import sol4_add as ad
from sol4_utils import *
#Flag allows reading of non-standard size images of type PNG
DRIV_KERNAL = np.array([[1], [0], [-1]]).astype(np.float32)
Y_AXIS = 1
X_AXIS = 2
K_FACTOR = 0.04
def harris_corner_detector(im):
    '''
    :param im: input image
    :return: a list of tupels Representing indexes Indicating corners in the Image
    '''
    deriv_x = deriv_in_axis(im, X_AXIS)
    deriv_y = deriv_in_axis(im, Y_AXIS)
    bl_deriv_2_x = blure_im(np.multiply(deriv_x, deriv_x).astype(np.float32), create_kernel(3))
    bl_deriv_2_y = blure_im(np.multiply(deriv_y, deriv_y).astype(np.float32), create_kernel(3))
    bl_deriv_x_y = blure_im(np.multiply(deriv_y, deriv_x).astype(np.float32), create_kernel(3))
    bl_deriv_y_x = blure_im(np.multiply(deriv_x, deriv_y).astype(np.float32), create_kernel(3))
    trace_mat = bl_deriv_2_x + bl_deriv_2_y
    det_mat = np.multiply(bl_deriv_2_x, bl_deriv_2_y) - np.multiply(bl_deriv_x_y, bl_deriv_y_x)
    R_matrix = det_mat - K_FACTOR * np.multiply(trace_mat, trace_mat)
    binary_im = ad.non_maximum_suppression(R_matrix)
    non_zero_mat = np.nonzero(binary_im)
    final_list = np.fliplr(np.array(list(np.transpose(non_zero_mat))))
    return final_list

def deriv_in_axis(im, axis):
    '''

    :param im:
    :param axis:
    :return:
    '''
    drive_kern = DRIV_KERNAL if axis == X_AXIS else np.transpose(DRIV_KERNAL)
    return convolve2d(im, drive_kern, mode='same').astype(np.float32)


def sample_descriptor(im, pos, desc_rad):
    '''

    :param im:
    :param pos:
    :param desc_rad:
    :return:
    '''
    disc_size = 2 * desc_rad + 1
    # resiz_po = np.array([[point[0]/4,point[1]/4] for point in pos])
    discriptor_met = np.zeros((disc_size,disc_size,len(pos)))
    for i,point in enumerate(pos):
        discriptor_corde = np.transpose(np.fliplr(creat_index_for_discriptor(point,desc_rad)))
        im_intens = ter.map_coordinates(im, discriptor_corde, order=1,prefilter=False)
        temp_calc = im_intens - np.average(im_intens)
        norm_dis = (((temp_calc) / np.linalg.norm(temp_calc)).astype(np.float32)).reshape((disc_size,disc_size))
        discriptor_met[:,:,i] = norm_dis
    return discriptor_met


def creat_index_for_discriptor(pos,desc_rad):
    '''

    :param pos:
    :param desc_rad:
    :return:
    '''
    nx = np.linspace(-desc_rad,desc_rad, 2 * desc_rad + 1)
    ny = np.linspace(-desc_rad, desc_rad, 2 * desc_rad + 1)
    base_disc_x,base_disc_y = np.meshgrid(nx,ny)
    discriptor = np.zeros(((2 * desc_rad + 1)**2, 2), dtype=np.float32)
    discriptor[:,0] = np.ndarray.flatten(base_disc_x + pos[0])
    discriptor[:,1] = np.ndarray.flatten(base_disc_y + pos[1])

    return discriptor


def find_features(pyr):
    pos = ad.spread_out_corners(pyr[0],7,7,12)
    # pos = harris_corner_detector(pyr[0])
    resize_point = pos/4
    desc = sample_descriptor(pyr[2], resize_point, 3)
    return pos, desc

def match_features(desc1,desc2,min_score):
    res_reshped_N1 = desc1.reshape(-1, desc1.shape[-1]).astype(np.float32).T
    res_reshped_N2 = (desc2.reshape(-1, desc2.shape[-1])).astype(np.float32)
    match_mat = (np.dot(res_reshped_N1,res_reshped_N2).astype(np.float32))
    match_mat[np.where(match_mat < min_score)] = 0
    bool_mat_x = np.zeros(match_mat.shape)
    bool_mat_y = np.zeros(match_mat.shape)
    most_like_by_N1 = (np.argsort(match_mat, axis=0)[-2:,:]).T
    most_like_by_N2 = np.argsort(match_mat, axis=1)[:, -2:]
    for i,j in enumerate(most_like_by_N2):
        if j[0] > min_score:
            bool_mat_x[i,j[0]] = 1
        if j[1] > min_score:
            bool_mat_x[i,j[1]] = 1
    for i,j in enumerate(most_like_by_N1):
        if j[0] > min_score:
            bool_mat_y[j[0],i] = 1
        if j[1] > min_score:
            bool_mat_y[j[1],i] = 1
    final_match = np.multiply(bool_mat_x.astype(np.float32),bool_mat_y.astype(np.float32))
    dis_match_N1, dis_match_N2 = np.where(final_match == 1)
    return dis_match_N1, dis_match_N2

def apply_homography(pos1, H12):
    three_cord = np.ones((pos1.shape[0],pos1.shape[1]+1))
    three_cord[:,:-1] = pos1
    three_cord = three_cord.T
    new_cords = np.dot(H12, three_cord).astype(np.float32)
    new_cords = new_cords / new_cords[2, :]
    final_mat = np.zeros((pos1.shape[1],pos1.shape[0]))
    final_mat[:,:]= new_cords[:-1,]
    return final_mat.T

def ransac_homography(pos1,pos2, num_iter, inliner_tol):
    '''

    :param pos1:
    :param pos2:
    :param num_iter:
    :param inliner_tol:
    :return:
    '''

    pos_4_1 = np.zeros((4,2))
    pos_4_2 = np.zeros((4,2))
    nx = np.arange(4).astype(np.int32)
    indices_for_rensac = np.arange(pos1.shape[0])
    final_inliners = np.array((0,0))
    while num_iter > 0:
        np.random.shuffle(indices_for_rensac)
        pos_4_1[nx] = pos1[indices_for_rensac[:4]]
        pos_4_2[nx] = pos2[indices_for_rensac[:4]]
        h12 = ad.least_squares_homography(pos_4_1, pos_4_2)
        if h12 is None:
            continue
        p_1_to_2 = apply_homography(pos1[:,:], h12.astype(np.float32)).astype(np.float32)
        error = np.linalg.norm(p_1_to_2[:,:] - pos2[:,:],axis=1)**2
        tamp = np.where(error < inliner_tol)

        final_inliners = tamp if np.size(tamp) > np.size(final_inliners) else final_inliners
        num_iter -= 1
    points1, points2 = pos1[final_inliners], pos2[final_inliners]
    final_h12 = ad.least_squares_homography(points2,points1)

    return final_h12, final_inliners


def display_matches(im1, im2, pos1, pos2, inliers):
    double_im = np.hstack((im1,im2))
    pos2[:,0] = pos2[:,0] + im1.shape[1]
    pos1_inliners = pos1[inliers]
    pos2_inliners = pos2[inliers]
    pos1[inliers, :] = 0
    pos1[inliers, :] = 0
    pos_outliers_idx = np.arange(pos1.shape[0])
    pos_outliers_idx = np.delete(pos_outliers_idx, inliers)
    pos1_outliers = pos1[pos_outliers_idx]
    pos2_outliers = pos2[pos_outliers_idx]
    plt.figure()
    # plt.imshow(double_im, plt.cm.gray)
    #
    # plt.plot((pos1_outliers[:,0],pos2_outliers[:,0]),(pos1_outliers[:,1],pos2_outliers[:,1]), mfc='r',
    #          c='b', lw=0.5, marker='.')
    # plt.plot((pos1_inliners[:,0],pos2_inliners[:,0]),(pos1_inliners[:,1],pos2_inliners[:,1]), mfc='r',
    #          c='y', lw=0.5, marker='.')
    # plt.show()


def accumulate_homographies2(H_successive,m):
    # H_successive.insert(m, np.eye(3))
    print("to be accumulated", H_successive)
    full_mat = [0]*(len(H_successive))
    full_mat.insert(m, np.eye(3))
    cur_mat = full_mat[m]
    for i in range(m-1,-1,-1):
        cur_mat = np.dot(cur_mat,H_successive[i])
        full_mat[i] = cur_mat
    cur_mat = full_mat[m]
    for j in range(m,len(H_successive),):
        cur_mat = np.dot(cur_mat, np.linalg.inv(H_successive[j]))
        full_mat[j+1] = cur_mat
    final_warp = [(mat / mat[2, 2]).astype(np.float32) for mat in full_mat]
    for i in final_warp:
        print (i)
    return final_warp

def accumulate_homographies(H_successive,m):
    H2m = [None] * (len(H_successive) + 1)
    H2m[m] = np.eye(3)

    for i in range(m - 1, -1, -1):
        H2m[i] = np.dot(H2m[i + 1], H_successive[i])

    for i in range(m + 1, len(H_successive) + 1):
        H2m[i] = np.dot(H2m[i - 1], np.linalg.inv(H_successive[i - 1]))
    final_warp = [(mat / mat[2, 2]).astype(np.float32) for mat in H2m]
    return final_warp

def get_canves_size(ims,Hs):
    cord_mat = np.zeros((len(ims),4,2))
    for i,img in enumerate(ims):
            h,w = img.shape
            pos = np.fliplr(np.array([[0,0],[h,0],[0,w],[h,w]]))
            # print(pos.shape)
            new_cord = apply_homography(pos,np.linalg.inv(Hs[i]))
            # print(new_cord)
            cord_mat[i,:,:] = new_cord

    w_max = np.ceil(np.max(cord_mat[:, :, 0])).astype(np.int32)
    w_min = np.floor(np.min(cord_mat[:, :, 0])).astype(np.int32)
    row_max = np.ceil(np.max(cord_mat[:, :, 1])).astype(np.int32)
    row_min = np.floor(np.min(cord_mat[:, :, 1])).astype(np.int32)
    # print(row_min, row_max, w_min, w_max)
    # plt.scatter(np.ndarray.flatten(cord_mat[0,:, 0]),np.ndarray.flatten(cord_mat[0,:,1]),marker="o",
    #             edgecolors="g")
    # plt.scatter(np.ndarray.flatten(cord_mat[1, :, 0]), np.ndarray.flatten(cord_mat[1, :, 1]),marker="*",
    #             edgecolors="b")
    # plt.scatter(np.ndarray.flatten(cord_mat[2, :, 0]), np.ndarray.flatten(cord_mat[2, :, 1]),marker="*",
    #             edgecolors="r")
    # plt.scatter(np.ndarray.flatten(cord_mat[3, :, 0]), np.ndarray.flatten(cord_mat[3, :, 1]),marker="o",
    #             edgecolors="y")
    # plt.show()
    return row_min, row_max,w_min,w_max, cord_mat


def get_boundris(ims,Hs):
    centers = []
    for i in range(len(ims)-1):
            h_1, w_1 = ims[i].shape
            h_2, w_2 = ims[i+1].shape
            center_im_1 = np.fliplr(np.array([[int(h_1//2),int(w_1//2)]]))
            center_im_2 = np.fliplr(np.array([[int(h_2//2),int(w_2//2)]]))
            new_center_1 = np.fliplr(apply_homography(center_im_1,np.linalg.inv(Hs[i])))
            new_center_2 = np.fliplr(apply_homography(center_im_2,np.linalg.inv(Hs[i+1])))
            centers.append(int((new_center_1[:,1] + new_center_2[:,1])//2))
    return centers



def render_panorama(ims,Hs):
    row_min, row_max, w_min, w_max ,cords = get_canves_size(ims, Hs)
    totel_w = w_max - w_min
    totle_h = row_max - row_min
    image_bound = [w_min]
    image_bound.extend(get_boundris(ims,Hs))
    image_bound.append(w_max)
    ny = np.linspace(row_min,row_max,totle_h).astype(np.int32)
    nx = np.linspace(w_min, w_max, totel_w).astype(np.int32)
    canves_x_ind, canves_y_ind = np.meshgrid(nx,ny)
    final_im = creat_canves(totle_h,totel_w)
    for i, im in enumerate(ims):
        im_left_bound = image_bound[i] + abs(w_min)
        im_right_bound = image_bound[i+1] + abs(w_min)
        # print("imge bound", (im_left_bound,im_right_bound))
        # extra_left = 0 if i == 0 else 128
        # extra_right = 0 if i == len(ims)-1 else 128
        # left_ind_final = (im_left_bound - extra_left)
        # right_ind_final = (im_right_bound + extra_right)
        im_size = im_right_bound - im_left_bound
        im_ind = creat_im_ind(canves_x_ind, canves_y_ind, [im_left_bound, im_size])
        print("sent to apply\n", im_ind.shape)
        warp_back_index = np.transpose(np.fliplr(apply_homography(im_ind, Hs[i])))
        print("eldan to interpolate",warp_back_index.shape)
        image_int = ter.map_coordinates(im, warp_back_index, order=1, prefilter=False)


        image_strip = (image_int.astype(np.float32)).reshape((totle_h, im_size))
        if i == 0:
            final_im[:, im_left_bound:im_right_bound] = image_strip
        else:
            final_im = blending_images(final_im,image_strip,im_left_bound,im_right_bound,totle_h,totel_w,im_left_bound)
    return final_im

def creat_im_ind(canve_cord_x, canve_cord_y, bounds):
    # print("eldan bounds", bounds[0],bounds[0]+bounds[1])
    x_cord = canve_cord_x[:,bounds[0]:bounds[0]+bounds[1]]
    y_cord = canve_cord_y[:,bounds[0]:bounds[0]+bounds[1]]
    indexs = np.zeros((np.size(x_cord), 2), dtype=np.float32)
    indexs[:,0] = np.ndarray.flatten(x_cord)
    indexs[:,1] = np.ndarray.flatten(y_cord)
    return indexs


def creat_canves(totle_h,totel_w):
    return np.zeros((totle_h, totel_w),dtype=np.float32)

def creat_binary_mask(left_ind_final,totle_h,totel_w):
    canves = creat_canves(totle_h,totel_w)
    canves[:, left_ind_final:] = canves[:, left_ind_final:] + 1
    return canves

def put_im_in_canves(totle_h,totel_w,left_ind_final,right_ind_final,im):
    canves_img = creat_canves(totle_h,totel_w)
    canves_img[:,left_ind_final:right_ind_final] = im
    return canves_img

def blending_images(final_im, im2,left_ind_final,right_ind_final,totle_h,totel_w,im_left_bound):
    mask = creat_binary_mask(im_left_bound,totle_h,totel_w)
    resize_image_strip = put_im_in_canves(totle_h,totel_w,left_ind_final,right_ind_final,im2)
    blanded_im = pyramid_blending(resize_image_strip,final_im,mask.astype(np.bool),14,3,3)
    return blanded_im




# if __name__ == '__main__':
#     im = read_image("backyard2.jpg",GRAY)
#     im2 = read_image("backyard3.jpg",GRAY)
#     du_im = np.hstack((im,im2))
#     gaus1 = build_gaussian_pyramid(im,3,3)[0]
#     gaus2 = build_gaussian_pyramid(im2,3,3)[0]
#     pos1,desc1 = find_features(gaus1)
#     pos2,desc2 = find_features(gaus2)
#     match_index1, mach_index2 = match_features(desc1,desc2,0.5)
#     points1, points2 = pos1[match_index1,:], pos2[mach_index2,:]
#     h12, inliners = ransac_homography(points1, points2, 1000, 6)
#     display_matches(im,im2,points1,points2,inliners)



# ny=np.linspace(-3,3,2*3+1)
# discriptoer_x,discrip_y = np.meshgrid(nx,ny)
# pos_4_1 = np.zeros((4,2))
# print(pos_4_1)
# a = np.array([[1,1],[2,5],[4,8],[3,4],[3,2]])
# print(a.shape)
# indices_for_rensac = np.arange(4)
# print(indices_for_rensac)
# np.random.shuffle(indices_for_rensac)
# print(indices_for_rensac)
# pos_4_1[nx] = a[indices_for_rensac[:4]]
# print(pos_4_1)
# nlarge_cord = np.array(a)
# print(nlarge_cord.shape)
# print(nlarge_cord)
# three_cord = nlarge_cord
# z_cord = np.ones((nlarge_cord.shape[0],nlarge_cord.shape[1]+1))
# z_cord = z_cord + 3
# z_cord[:,:-1]= three_cord
# print(z_cord.shape)
# print(z_cord)
# z_cord = (z_cord).T
# print(z_cord.shape)
# print(z_cord)
# z_cord = z_cord / z_cord[2,:]
# print(z_cord.shape)
# print(z_cord)
# final_mat = np.zeros((nlarge_cord.shape[1],nlarge_cord.shape[0]))
# final_mat[:,:]= z_cord[:-1,:]
# print(final_mat)
# b = np.zeros(discriptoer_x.shape)
# print(discriptoer_x)
# print(np.argsort(discriptoer_x,axis=0))
# # b[] = 1
# print(b)
# # r[np.where(r>0.5)]=0
# print(r)
# print(r.shape)
# r = r.reshape(-1,r.shape[-1])
# print(r.shape)
# print(np.argmax(match_mat[2,:]))
# r[:,:,0] = discriptoer_x
# r[:,:,1] = discrip_y
# print(r)