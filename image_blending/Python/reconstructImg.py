'''
  File name: reconstructImg.py
  Author:
  Date created:
'''

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from numpy.linalg import inv
from scipy import signal


def reconstructImg(indexes, red, green, blue, targetImg):
    resultImg = np.copy(targetImg).astype(int)
    maxi = int(np.amax(indexes))
    for x in range(maxi):
        i,j = np.where(indexes==x+1)
        resultImg[i,j,0] = red[x]
        resultImg[i,j,1] = green[x]
        resultImg[i,j,2] = blue[x]
    resultImg = np.clip(resultImg,0,255)
    return resultImg