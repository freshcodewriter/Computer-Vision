"""
  File name: mymosaic.py
  Author:
  Date created:
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from ransac_est_homography import ransac_est_homography

thresh = 0.5

img_m = mpimg.imread('middle.jpg')
img_l = mpimg.imread('left.jpg')
NUM_channels = 3


def show_ransac():
    big_im = np.concatenate((img_l, img_m), axis=1)

    plt.imshow(big_im)

    x2_shift = x2 + img_m.shape[1]
    for i in range(x1.shape[0]):
        if inline_idx[i] == 1:
            plt.plot([x1[i], x2_shift[i]], [y1[i], y2[i]], marker="o")
    plt.show()


if __name__ == "__main__":

    x1 = np.load('./debug-1106/draw_x1.npy')
    y1 = np.load('./debug-1106/draw_y1.npy')
    x2 = np.load('./debug-1106/draw_x2.npy')
    y2 = np.load('./debug-1106/draw_y2.npy')
    print('x1:\n', x1)
    print('y1:\n', y1)
    print('x2:\n', x2)
    print('y2:\n', y2)
    ransac_H, inline_idx = ransac_est_homography(x1, x2, y1, y2, 0.5)
    print("inline_idx:\n", inline_idx)
    print("ransac_H:\n", ransac_H)
    show_ransac()


