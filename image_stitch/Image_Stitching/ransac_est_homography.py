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
import scipy
import matplotlib.pyplot as plt
import math
import sys
import warnings
warnings.filterwarnings('error')
np.seterr(all='warn')
num_ransac = 1000
min_consensus = 0 # Todo: 10 will leave nothing

debug_i = 0


def random_sample(x1, y1, x2, y2):
    # Randomly select four feature pairs
    # points are unique within each sample; could be repeated between samples
    sample_idx = np.random.choice(x1.shape[0], 4)
    # print('---random_sample---')
    # print('sample_idx:', sample_idx)
    # print(x1[sample_idx], y1[sample_idx], x2[sample_idx], y2[sample_idx])
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
        print('-------- Get Warning! --------')
        print('x:', x)
        print('y:', y)
        print('affine_src:\n', affine_src)
        print('H:\n', H)
        print('affine_target:\n', affine_target)
    return affine_target


def get_ssd(x1, y1, x2, y2, H):
    affine_1 = apply_homography(x1, y1, H)
    affine_2 = get_affine_matrix(x2, y2)
    # Compute SSD between transformed source and target
    ssd = np.sum((np.square(affine_1 - affine_2)), axis=0)
    return ssd


def ransac_est_homography(x1, y1, x2, y2, thresh):
    print('----------- ransac_est_homography -----------')
    print('[ransac_est_homography] total number of points: ', x1.shape[0])
    current_most_inline = min_consensus
    ransac_H = np.zeros((3, 3))
    inline_idx = np.zeros(x1.shape[0]).astype(int)
    ransac_inline_idx = inline_idx.copy()
    for i in range(num_ransac):
        debug_i = i
        sample_x1, sample_y1, sample_x2, sample_y2 = random_sample(x1, y1, x2, y2)
        sample_H = est_homography(sample_x1, sample_y1, sample_x2, sample_y2)
        # print('sample_H:\n', sample_H)
        ssd = get_ssd(x1, y1, x2, y2, sample_H)
        # print('ssd:', ssd)
        inline_idx = np.zeros(x1.shape[0]).astype(int)
        inline_idx[np.where(ssd < thresh)] = 1

        num_inline = np.count_nonzero(inline_idx)

        if num_inline > current_most_inline:
            print('update!')
            print('num_inline:', num_inline)
            print('ssd:\n', ssd)
            ransac_H = sample_H.copy()
            current_most_inline = num_inline
            ransac_inline_idx = inline_idx.copy()
        # exit()
    return ransac_H, ransac_inline_idx

