"""
  File name: mymosaic.py
  Author:
  Date created:
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from final_feature_match import feature_matching

from mosaic_helper import draft, wrap_img, stitch, get_imgs
from ransac_est_homography import ransac_est_homography
from seam import seam

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. 
    If you want to implement imwarp (or similar function) by yourself, 
    you should apply bilinear interpolation when you copy pixel values. 
    
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input img_input: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''

thresh = 10
NUM_channels = 3


def stitching_l(img_input, H):
    img_wrap_left = img_input[0]
    img_ref = img_input[1]
    h, w, _ = img_ref.shape
    H_left = H

    # wrap -> reference
    def calculate_overlap(ref_ori, wrap_ori, img_wrap, canvas):
        print('----- calculate_overlap -----')
        # print('ref_ori:', ref_ori)
        # print('wrap_ori:', wrap_ori)
        # print('img_wrap.shape', img_wrap.shape)
        # exit()
        overlap_x = [ref_ori[0], img_wrap.shape[1] - 1]
        if ref_ori[1] > 0:
            overlap_y = [ref_ori[1], min(img_wrap.shape[0], ref_ori[1] + h)]
        else:
            overlap_y1 = wrap_ori[1]
            if img_wrap.shape[0] + wrap_ori[1] < canvas.shape[1]:
                overlap_y2 = img_wrap.shape[0] + wrap_ori[1] - 1  # TODO: wired 1 offset
            else:
                overlap_y2 = h
            overlap_y = [overlap_y1, overlap_y2]
        return overlap_x, overlap_y

    def overlapping_handler(overlap_x, overlap_y, canvas, res_wrap, res_ref):
        print('---- overlapping_handler ----')
        print(canvas.shape)
        print(overlap_x)
        print(overlap_y)

        overlap_pts_x = [overlap_x[0], overlap_x[0], overlap_x[1], overlap_x[1], overlap_x[0]]
        overlap_pts_y = [overlap_y[0], overlap_y[1], overlap_y[1], overlap_y[0], overlap_y[0]]

        overlap_pts_mask = np.zeros_like(canvas[:, :, 0])
        overlap_pts_mask[overlap_pts_y, overlap_pts_x] = 1

        carv_img_wrap = res_wrap[overlap_y[0]: overlap_y[1] + 1, 0: overlap_x[1] + 1, :]
        overlap_pts_mask_wrap = overlap_pts_mask.copy()[overlap_y[0]: overlap_y[1] + 1, 0: overlap_x[1] + 1]

        carv_img_ref = res_ref[overlap_y[0]: overlap_y[1] + 1, overlap_x[0]::, :]
        overlap_pts_mask_ref = overlap_pts_mask.copy()[overlap_y[0]: overlap_y[1] + 1, overlap_x[0]::]

        pts_y, pts_x = np.where(overlap_pts_mask_wrap == 1)
        y1 = y4 = pts_y[0]
        y2 = y3 = pts_y[2]
        x1 = x2 = pts_x[0]
        x3 = x4 = pts_x[1]
        overlap_boundary_pts_wrap = np.asarray([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        pts_y, pts_x = np.where(overlap_pts_mask_ref == 1)
        y1 = y4 = pts_y[0]
        y2 = y3 = pts_y[2]
        x1 = x2 = pts_x[0]
        x3 = x4 = pts_x[1]
        overlap_boundary_pts_ref = np.asarray([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        return carv_img_wrap, carv_img_ref, overlap_boundary_pts_wrap, overlap_boundary_pts_ref

    print('Stitching left...')
    wrap_boundary_l = draft(img_ref, H_left)
    img_wrap = wrap_img(img_wrap_left, wrap_boundary_l, H_left)
    canvas_left, ref_ori_left, wrap_ori_left = stitch(img_wrap, img_ref, wrap_boundary_l, w, h)
    # Put wrap and ref images in the same canvas with the result of stitch
    [canvas_ref, canvas_wrap] = get_imgs(img_wrap, img_ref, wrap_boundary_l)
    # Subset wrap and ref by overlapping area
    overlap_x, overlap_y = calculate_overlap(ref_ori_left, wrap_ori_left, img_wrap, canvas_left)
    plt.imshow(canvas_left)
    plt.show()
    # exit()
    carv_img_wrap, carv_img_ref, overlap_b_pts_wrap, overlap_b_pts_ref = overlapping_handler(overlap_x, overlap_y, canvas_left, canvas_wrap, canvas_ref)
    # exit()
    result = seam(carv_img_wrap, carv_img_ref, overlap_b_pts_wrap, overlap_b_pts_ref)
    return result


def stitching_r(img_input, H):
    img_ref = img_input[0]
    img_wrap_right = img_input[1]
    h, w, _ = img_ref.shape
    H_right = H

    # Here wrap is the right image
    # reference <- wrap
    def calculate_overlap(ref_ori, wrap_ori, img_wrap, canvas):
        overlap_x = [wrap_ori[0], w - 1]
        if ref_ori[1] > 0:
            overlap_y = [ref_ori[1], min(img_wrap.shape[0], ref_ori[1] + h)]
        else:  # ref_ori[1] = 0
            overlap_y1 = wrap_ori[1]
            if img_wrap.shape[0] + wrap_ori[1] < canvas.shape[1]:
                overlap_y2 = img_wrap.shape[0] + wrap_ori[1] - 1  # TODO: wired 1 offset
            else:
                overlap_y2 = h
            overlap_y = [overlap_y1, overlap_y2]
        return overlap_x, overlap_y

    # ref is the left image Wrap is the right image
    def overlapping_handler(overlap_x, overlap_y, canvas, res_wrap, res_ref):
        overlap_pts_x = [overlap_x[0], overlap_x[0], overlap_x[1], overlap_x[1], overlap_x[0]]
        overlap_pts_y = [overlap_y[0], overlap_y[1], overlap_y[1], overlap_y[0], overlap_y[0]]

        overlap_pts_mask = np.zeros_like(canvas[:, :, 0])
        overlap_pts_mask[overlap_pts_y, overlap_pts_x] = 1

        carv_img_wrap = res_wrap[overlap_y[0]: overlap_y[1] + 1, overlap_x[0]::, :]
        overlap_pts_mask_wrap = overlap_pts_mask.copy()[overlap_y[0]: overlap_y[1] + 1, overlap_x[0]::]
        carv_img_ref = res_ref[overlap_y[0]: overlap_y[1] + 1, 0: overlap_x[1] + 1, :]
        overlap_pts_mask_ref = overlap_pts_mask.copy()[overlap_y[0]: overlap_y[1] + 1, 0: overlap_x[1] + 1]

        pts_y, pts_x = np.where(overlap_pts_mask_wrap == 1)
        y1 = y4 = pts_y[0]
        y2 = y3 = pts_y[2]
        x1 = x2 = pts_x[0]
        x3 = x4 = pts_x[1]
        overlap_boundary_pts_wrap = np.asarray([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        pts_y, pts_x = np.where(overlap_pts_mask_ref == 1)
        y1 = y4 = pts_y[0]
        y2 = y3 = pts_y[2]
        x1 = x2 = pts_x[0]
        x3 = x4 = pts_x[1]
        overlap_boundary_pts_ref = np.asarray([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        return carv_img_wrap, carv_img_ref, overlap_boundary_pts_wrap, overlap_boundary_pts_ref

    wrap_boundary_r = draft(img_ref, H_right)
    img_wrap = wrap_img(img_wrap_right, wrap_boundary_r, H_right)
    # Wrap is the right image; Ref is the middle image
    canvas_right, wrap_ori_right, ref_ori_right = stitch(img_wrap, img_ref, wrap_boundary_r, w, h)
    [canvas_ref, canvas_wrap] = get_imgs(img_wrap, img_ref, wrap_boundary_r)
    # Switch order here
    overlap_x, overlap_y = calculate_overlap(wrap_ori_right, ref_ori_right, img_wrap, canvas_right)
    carv_img_wrap, carv_img_ref, overlap_b_pts_wrap, overlap_b_pts_ref = overlapping_handler(overlap_x, overlap_y, canvas_right, canvas_wrap, canvas_ref)
    result = seam(carv_img_ref, carv_img_wrap, overlap_b_pts_ref, overlap_b_pts_wrap)
    return result


def show_ransac(img_wrap, img_ref, x2, y2, x1, y1, inline_idx):
    big_im = np.concatenate((img_ref, img_wrap), axis=1)
    plt.imshow(big_im)
    x2_shift = x2 + img_ref.shape[1]
    for i in range(x1.shape[0]):
        if inline_idx[i] == 1:
            plt.plot([x1[i], x2_shift[i]], [y1[i], y2[i]], marker="o")
    plt.show()


def remove_edge(canvas):
    row_sum_0 = np.sum(canvas[:, :, 0], axis=1)
    row_sum_1 = np.sum(canvas[:, :, 1], axis=1)
    row_sum_2 = np.sum(canvas[:, :, 2], axis=1)
    row_sum = row_sum_0 + row_sum_1 + row_sum_2
    canvas = canvas[row_sum > 0]

    col_sum_0 = np.sum(canvas[:, :, 0], axis=0)
    col_sum_1 = np.sum(canvas[:, :, 1], axis=0)
    col_sum_2 = np.sum(canvas[:, :, 2], axis=0)
    col_sum = col_sum_0 + col_sum_1 + col_sum_2
    return canvas[:, col_sum > 0, :]


def mymosaic(img_input):
    img_wrap_l = img_input[0]
    img_ref = img_input[1]
    img_wrap_r = img_input[2]
    x_ref_left, y_ref_left, x_wrap_left, y_wrap_left = feature_matching(img_ref, img_wrap_l, 'left')
    ransac_H_left, inline_idx_left = ransac_est_homography(x_wrap_left, y_wrap_left, x_ref_left, y_ref_left, thresh)
    canvas = stitching_l([img_wrap_l, img_ref], ransac_H_left)
    x_ref_right, y_ref_right, x_wrap_right, y_wrap_right = feature_matching(canvas, img_wrap_r, 'right')
    ransac_H_right, inline_idx_right = ransac_est_homography(x_wrap_right, y_wrap_right,
                                                             x_ref_right, y_ref_right, thresh)
    canvas = stitching_r([canvas, img_wrap_r], ransac_H_right)
    canvas = remove_edge(canvas)
    return canvas

