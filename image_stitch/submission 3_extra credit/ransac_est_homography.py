"""
  File name: ransac_est_homography.py
  Author:
  Date created:
"""
from est_homography import est_homography
import pprint as pp
'''
  File clarification: Use a robust method (RANSAC) to compute a homography. 
  Use 4-point RANSAC as described in class to compute a robust homography estimate: 
  - Input x1, y1, x2, y2: 
      N × 1 vectors representing the correspondences feature coordinates 
      in the first and second image. 
      It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image. 
  - Input thresh: 
      the threshold on distance used to determine if transformed points agree. 
  - Output H: 
      3 × 3 matrix representing the homograph matrix computed in final step of RANSAC. 
  - Output inlier_ind: 
      N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier. 
'''
import numpy as np
import warnings
warnings.filterwarnings('error')
np.seterr(all='warn')
num_ransac = 1000
debug_i = 0


def random_sample(x1, y1, x2, y2):
    # Randomly select four feature pairs
    # points are unique within each sample; could be repeated between samples
    sample_idx = np.random.choice(x1.shape[0], 4)
    return x1[sample_idx], y1[sample_idx], x2[sample_idx], y2[sample_idx]


def get_affine_matrix(x, y):
    n = x.shape[0]
    affine_matrix = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), np.ones((n, 1))), axis=1).transpose((1, 0))
    return affine_matrix


def apply_homography(x, y, H):
    # ([X, Y, 1] ^ T ~ H *[x, y, 1] ^ T)
    affine_src = get_affine_matrix(x, y)
    affine_target = np.matmul(H, affine_src)
    # Normalize the last row
    # affine_target = np.divide(affine_target, affine_target[2, :])
    # TODO: do not choose points too close to edge
    try:
        affine_target = np.divide(affine_target, affine_target[2, :])
    except Warning:
        pass
        # print('-------- Get Warning! --------')
        # print('x:', x)
        # print('y:', y)
        # print('affine_src:\n', affine_src)
        # print('H:\n', H)
        # print('affine_target:\n', affine_target)
    return affine_target


def get_ssd(x_wrap, y_wrap, x_ref, y_ref, H):
    affine_1 = apply_homography(x_wrap, y_wrap, H)
    affine_2 = get_affine_matrix(x_ref, y_ref)
    # Compute SSD between transformed source and target
    ssd = np.sum((np.square(affine_1 - affine_2)), axis=0)
    return ssd


def ransac_est_homography(x_wrap, y_wrap, x_ref, y_ref, thresh):
    # print('----------- ransac_est_homography -----------')
    # print('[ransac_est_homography] total number of points: ', x1.shape[0])
    current_most_inline = 0
    inline_idx = np.zeros(x_wrap.shape[0]).astype(int)
    ransac_inline_idx = inline_idx.copy()

    for i in range(num_ransac):
        sample_x_wrap, sample_y_wrap, sample_x_ref, sample_y_ref = random_sample(x_wrap, y_wrap, x_ref, y_ref)
        # Ref = H * Wrap
        sample_H = est_homography(sample_x_wrap, sample_y_wrap, sample_x_ref, sample_y_ref)
        # print('sample_H:\n', sample_H)
        ssd = get_ssd(x_wrap, y_wrap, x_ref, y_ref, sample_H)
        # print('ssd:', ssd)
        inline_idx = np.zeros(x_wrap.shape[0]).astype(int)
        inline_idx[np.where(ssd < thresh)] = 1
        num_inline = np.count_nonzero(inline_idx)

        if num_inline > current_most_inline:
            current_most_inline = num_inline
            ransac_inline_idx = inline_idx.copy()

    ransac_H = est_homography(x_wrap[ransac_inline_idx == 1],
                              y_wrap[ransac_inline_idx == 1],
                              x_ref[ransac_inline_idx == 1],
                              y_ref[ransac_inline_idx == 1])
    # print('num_inline: ', num_inline)
    return ransac_H, ransac_inline_idx

