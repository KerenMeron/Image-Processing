#############################################################
# FILE :sol4.py
# WRITER : Eldan chodorov , eldan , 201335965
# EXERCISE : image Proc sol4 2016-2017
#############################################################
from scipy.ndimage import interpolation as ter


import sol4_add as ad
from sol4_utils import *
#Flag allows reading of non-standard size images of type PNG
DRIV_KERNAL = np.array([[1], [0], [-1]]).astype(np.float32)
Y_AXIS = 1
X_AXIS = 2
K_FACTOR = 0.04
RESIZE_FACTOR = 4
def harris_corner_detector(im):
    '''
    The method is responsible for identifying locations in the image that have a unique Detection like
    a corner of an object
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
    The method is responsible for creating a derivative of an image of a particular axis
    returns the deriv image
    :param im:
    :param axis: axis, whereby the method generates the derived image
    Y_AXIS = 1
    X_AXIS = 2
    '''
    drive_kern = DRIV_KERNAL if axis == X_AXIS else np.transpose(DRIV_KERNAL)
    return convolve2d(im, drive_kern, mode='same').astype(np.float32)


def sample_descriptor(im, pos, desc_rad):
    '''
    Produces a matrix descriptor, describes the point identified as interesting,
    by sampling the strengths of her surroundings from the third stage of the
    pyramid image return a descriptor matrix
    :param desc_rad: the radius of the Matrix
    '''
    disc_size = 2 * desc_rad + 1
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
    Creates a matrix of indexes surrounding the index Interest
    :param pos:
    :param desc_rad:
    :return: a vector of two-dimensions, the index of X and Y
    '''
    nx = np.linspace(-desc_rad,desc_rad, 2 * desc_rad + 1)
    ny = np.linspace(-desc_rad, desc_rad, 2 * desc_rad + 1)
    base_disc_x,base_disc_y = np.meshgrid(nx,ny)
    discriptor = np.zeros(((2 * desc_rad + 1)**2, 2), dtype=np.float32)
    discriptor[:,0] = np.ndarray.flatten(base_disc_x + pos[0])
    discriptor[:,1] = np.ndarray.flatten(base_disc_y + pos[1])

    return discriptor


def find_features(pyr):
    '''
    Gtting pyramid image and identifies points of interest in the image
    At the end returns a list of indexes and discriptor of the points of the  image
    :param pyr:
    :return:
    '''
    pos = ad.spread_out_corners(pyr[0],7,7,12)
    # pos = harris_corner_detector(pyr[0])
    resize_point = pos/RESIZE_FACTOR
    desc = sample_descriptor(pyr[2], resize_point, 3).astype(np.float32)
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
    new_cords = np.nan_to_num(np.dot(H12, three_cord).astype(np.float32))
    new_cords = new_cords / new_cords[2, :]
    final_mat = np.zeros((pos1.shape[1],pos1.shape[0]))
    final_mat[:,:]= new_cords[:-1,]
    return np.nan_to_num(final_mat.T)

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
        pos_4_1[nx] = (pos1[indices_for_rensac[:4]])
        pos_4_2[nx] = (pos2[indices_for_rensac[:4]])
        h12 = ad.least_squares_homography(pos_4_1, pos_4_2)
        if h12 is None:
            continue
        p_1_to_2 = apply_homography(pos1[:,:], h12.astype(np.float32)).astype(np.float32)
        error = np.linalg.norm(p_1_to_2[:,:] - pos2[:,:],axis=1)
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
    plt.imshow(double_im, plt.cm.gray)

    plt.plot((pos1_outliers[:,0],pos2_outliers[:,0]),(pos1_outliers[:,1],pos2_outliers[:,1]), mfc='r',
             c='b', lw=0.5, marker='.')
    plt.plot((pos1_inliners[:,0],pos2_inliners[:,0]),(pos1_inliners[:,1],pos2_inliners[:,1]), mfc='r',
             c='y', lw=0.5, marker='.')
    plt.show()


def accumulate_homographies(H_successive,m):
    full_mat = [0]*(len(H_successive))
    full_mat.insert(m, np.eye(3))
    cur_mat = full_mat[m]
    for i in range(m-1,-1,-1):
        cur_mat = np.dot(cur_mat,H_successive[i])
        full_mat[i] = cur_mat/cur_mat[2,2]
    cur_mat = full_mat[m]
    for j in range(m,len(H_successive)):
        cur_mat = np.dot(cur_mat, np.linalg.inv(H_successive[j]))
        full_mat[j+1] = cur_mat/cur_mat[2,2]
    return full_mat

# def accumulate_homographies2(H_successive, m):
#     H2m = [None] * (len(H_successive) + 1)
#     H2m[m] = np.eye(3)
#
#     for i in range(m - 1, -1, -1):
#         H2m[i] = np.dot(H2m[i + 1], H_successive[i])
#         H2m[i] = H2m[i]/H2m[i][2,2]
#
#     for i in range(m + 1, len(H_successive) + 1):
#         H2m[i] = np.dot(H2m[i - 1], np.linalg.inv(H_successive[i - 1]))
#         H2m[i] = H2m[i]/H2m[i][2,2]
#
#     return H2m

def get_canves_size(ims,Hs):
    cord_mat = np.zeros((len(ims),4,2))
    for i,img in enumerate(ims):
            h,w = img.shape
            pos = np.fliplr(np.array([[0,0],[h,0],[0,w],[h,w]]))
            new_cord = apply_homography(pos,np.linalg.inv(Hs[i]))
            cord_mat[i,:,:] = new_cord
    w_max = np.ceil(np.max(cord_mat[:, :, 0])).astype(np.int32)
    w_min = np.floor(np.min(cord_mat[:, :, 0])).astype(np.int32)
    row_max = np.ceil(np.max(cord_mat[:, :, 1])).astype(np.int32)
    row_min = np.floor(np.min(cord_mat[:, :, 1])).astype(np.int32)
    return row_min, row_max,w_min,w_max, cord_mat


def get_boundris(ims,Hs,w_min,w_max):
    centers = [w_min]
    for i in range(len(ims)-1):
            h_1, w_1 = ims[i].shape
            h_2, w_2 = ims[i+1].shape
            center_im_1 = np.fliplr(np.array([[int(h_1//2),int(w_1//2)]]))
            center_im_2 = np.fliplr(np.array([[int(h_2//2),int(w_2//2)]]))
            new_center_1 = np.fliplr(apply_homography(center_im_1,np.linalg.inv(Hs[i])))
            new_center_2 = np.fliplr(apply_homography(center_im_2,np.linalg.inv(Hs[i+1])))
            centers.append(int((new_center_1[:,1] + new_center_2[:,1])//2))
    centers.append(w_max)
    return centers

def creat_panorame_index(row_min, row_max, w_min, w_max):
    totel_w = w_max - w_min
    totle_h = row_max - row_min
    ny = np.linspace(row_min,row_max,totle_h).astype(np.int32)
    nx = np.linspace(w_min, w_max, totel_w).astype(np.int32)
    return np.meshgrid(nx,ny)

def render_panorama(ims,Hs):
    print("ELDAN HS\n\n", Hs)
    row_min, row_max, w_min, w_max ,cords = get_canves_size(ims, Hs)
    totel_w = w_max - w_min
    totle_h = row_max - row_min
    canves_bound = get_boundris(ims, Hs, w_min, w_max)
    canves_x_ind, canves_y_ind = creat_panorame_index(row_min,row_max,w_min,w_max)
    final_im = creat_canves(totle_h,totel_w)
    canves_bound = canves_bound - w_min
    for i, im in enumerate(ims):
        extand_left = 0 if i == 0 else 128
        extand_right = 0 if i == len(ims)-1 else 128
        l_canv_indx = (canves_bound[i] - extand_left)
        r_canv_indx = (canves_bound[i+1] + extand_right)
        im_size = r_canv_indx - l_canv_indx
        pano_ind = creat_pano_index(canves_x_ind, canves_y_ind, [l_canv_indx, im_size])
        warp_back_indx = np.transpose(np.fliplr(apply_homography(pano_ind,Hs[i])))
        image_int = ter.map_coordinates(im, warp_back_indx, order=1, prefilter=False)
        image_strip = (image_int.astype(np.float32)).reshape((totle_h, im_size))
        if i == 0:
            final_im[:, l_canv_indx:r_canv_indx] = image_strip
        else:
            final_im = blending_images(final_im, image_strip,l_canv_indx,r_canv_indx,totle_h,totel_w,
                                       canves_bound[i])
    return final_im

def creat_pano_index(canve_cord_x, canve_cord_y, bounds):
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
    resize_image_strip = put_im_in_canves(totle_h,totel_w,left_ind_final,right_ind_final, im2)
    blanded_im = pyramid_blending(resize_image_strip,final_im,mask.astype(np.bool),14,7,7)
    return blanded_im




