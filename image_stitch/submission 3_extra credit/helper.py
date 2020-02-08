
# coding: utf-8

# In[ ]:

import cv2
import numpy as np


def helper(x,y):
    x = x.reshape(x.shape[0],1)
    y = y.reshape(y.shape[0],1)
    xy = np.concatenate((x, y), axis=1)
    keypoints = [cv2.KeyPoint(x[0], x[1], 1) for x in xy]
    return xy,keypoints

