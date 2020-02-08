'''
  File name: feat_desc.py
  Author:
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature, 
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40 
    window to have a nice big blurred descriptor. 
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''
import numpy as np
from scipy import signal
from utils import GaussianPDF_2D
from utils import rgb2gray
from PIL import Image
from utils import visDerivatives

def findDerivatives(I_gray): 
    G = GaussianPDF_2D(0, 1, 5, 5)
    dx, dy = np.gradient(G, axis = (1,0))
    Magx = signal.convolve2d(I_gray,dx,'same')
    Magy = signal.convolve2d(I_gray,dy,'same')
    Mag = np.sqrt(Magx*Magx+Magy*Magy)
    Ori = np.arctan2(Magy, Magx)
    return Mag, Ori

def feat_desc(img, x, y):
    Mag, Ori = findDerivatives(img)
    des_size = 64
    des = np.zeros([des_size,x.shape[0]])
    dummy_patch = np.arange(-20,20,5)
    dummy_space = np.arange(0,5)
    space_x,space_y = np.meshgrid(dummy_space,dummy_space)
    patch_x, patch_y = np.meshgrid(dummy_patch,dummy_patch)
    patch_x, patch_y = patch_x.flatten(), patch_y.flatten()

    for j in range(x.shape[0]):
        for i in range(patch_x.shape[0]):
            patch_top_x = x[j]+patch_x[i]
            patch_top_y = y[j]+patch_y[i]

            patch_space_x = np.full((5, 5), patch_top_x)
            patch_space_x = patch_space_x + space_x

            patch_space_y = np.full((5, 5), patch_top_y)
            patch_space_y = patch_space_y + space_y

            patch_space_y = np.clip(patch_space_y, 0, img.shape[0] - 1).astype(np.int32)
            patch_space_x = np.clip(patch_space_x, 0, img.shape[1] - 1).astype(np.int32)

    #         print(patch_space_x)
    #         print(patch_space_y)
        #     mag_matrix = Mag_1[patch_space_y,patch_space_x]
        #     mag_max = np.amax(mag_matrix)
            mag_max = np.amax(Mag[patch_space_y,patch_space_x])
            des[i,j] = mag_max

    for i in range(0, des.shape[1]):
        des[:,i] = des[:,i] - np.mean(des[:,i])
        des[:,i] = des[:,i] / np.std(des[:,i])
    return des