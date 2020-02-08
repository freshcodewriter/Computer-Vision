'''
  File name: morph_tri.py
  Author:
  Date created:
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''

import numpy as np
from scipy.spatial import Delaunay
from interp import interp2
import matplotlib.pyplot as plt
import matplotlib.tri as t
import imageio

def getCorners(points, index, grid):
    Ax = points[index[grid][:,0]][:,0]
    Ay = points[index[grid][:,0]][:,1]
    Bx = points[index[grid][:,1]][:,0]
    By = points[index[grid][:,1]][:,1]
    Cx = points[index[grid][:,2]][:,0]
    Cy = points[index[grid][:,2]][:,1]
    return Ax, Bx, Cx, Ay, By, Cy

def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):

    # 1. compute the inter_shape_pts by taking average of two pictures
    inter_shape_pts = (im1_pts + im2_pts) / 2

    # 2. Triangulation using scipy.spatial.Delaunay
    tri = Delaunay(inter_shape_pts) 
    indices = tri.simplices 

    # make a mesh grid
    nr = im1.shape[0];
    nc = im1.shape[1];
    row, col = np.meshgrid(np.arange(nr), np.arange(nc))

    # store all images
    morphed_im = np.empty((warp_frac.shape[0], im1.shape[0], im1.shape[1], im1.shape[2])) 

    for i in range(len(warp_frac)):
        warp_im = im1_pts * (1-warp_frac[i]) + im2_pts * warp_frac[i]

        # correlates the current warp image to the average(.5) triangulation
        warp_tri = t.Triangulation(warp_im[:,0], warp_im[:,1], indices)

        find_triangle = warp_tri.get_trifinder() # find_triangle(x,y)

        # mesh grid - stores the triangle index of each pixel
        m_grid = find_triangle(row,col) 
        m_grid = m_grid.flatten() # make it 1D
        X_f = row.flatten()
        Y_f = col.flatten()
        
        # solve the barycentric coordinate for alpha, beta and gamma
        index = np.array(tri.simplices)
        Ax, Bx, Cx, Ay, By, Cy = getCorners(warp_im, index, m_grid)
        inverse = 1.0 /(Ax * By - Ax * Cy - Bx * Ay + Bx * Cy + Cx * Ay - Cx * By)
        alpha = inverse * ((By - Cy) * X_f + (Cx - Bx) * Y_f + Bx * Cy - Cx * By)
        beta = inverse * ((Cy - Ay) * X_f + (Ax - Cx) * Y_f + Cx * Ay - Ax * Cy)
        gamma = inverse * ((Ay - By) * X_f + (Bx - Ax) * Y_f + Ax * By - Bx * Ay)

        # get coordinate of source image
        Ax_s, Bx_s, Cx_s, Ay_s, By_s, Cy_s = getCorners(im1_pts, index, m_grid)
        x_im1 = Ax_s * alpha + Bx_s * beta + Cx_s * gamma
        y_im1 = Ay_s * alpha + By_s * beta + Cy_s * gamma

        # get coordinate of target image
        Ax_t, Bx_t, Cx_t, Ay_t, By_t, Cy_t = getCorners(im2_pts, index, m_grid)
        x_im2 = Ax_t * alpha + Bx_t * beta + Cx_t * gamma
        y_im2 = Ay_t * alpha + By_t * beta + Cy_t * gamma

        # get x,y coordinates using interp2 
        im1_red = im1[:,:,0]; im1_green = im1[:,:,1]; im1_blue = im1[:,:,2]
        im1_R = interp2(im1_red, x_im1, y_im1); im1_G = interp2(im1_green, x_im1, y_im1); im1_B = interp2(im1_blue, x_im1, y_im1)
        
        im2_red = im2[:,:,0]; im2_green = im2[:,:,1]; im2_blue = im2[:,:,2]
        im2_R = interp2(im2_red, x_im2, y_im2); im2_G = interp2(im2_green, x_im2, y_im2); im2_B = interp2(im2_blue, x_im2, y_im2)

        # # cross-dissolve, warp image1 rbg, image2 rgb
        red = im1_R * (1-dissolve_frac[i]) + im2_R * dissolve_frac[i]
        green = im1_G * (1-dissolve_frac[i]) + im2_G * dissolve_frac[i]
        blue = im1_B * (1-dissolve_frac[i]) + im2_B * dissolve_frac[i]

        red = red.reshape(nc, nr).astype(int)
        green = green.reshape(nc, nr).astype(int)
        blue = blue.reshape(nc, nr).astype(int)

        # store resulting image 
        morphed_im[i, :, :, 0] = red
        morphed_im[i, :, :, 1] = green
        morphed_im[i, :, :, 2] = blue
        morphed_im = morphed_im.astype(np.uint8)

        morphed_im_list = []
        
    for idx in range(morphed_im.shape[0]):
        morphed_im_list.append(morphed_im[idx, :, :, :])

    # generate gif file
    imageio.mimsave('./result/result.gif', morphed_im_list)

    return morphed_im
