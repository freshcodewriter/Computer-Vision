'''
  File name: getCoefficientMatrix.py
  Author:
  Date created:
'''
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from numpy.linalg import inv
from scipy import signal

def getCoefficientMatrix(indexes):
    maxi = int(np.amax(indexes))
    coeffA = np.zeros((maxi,maxi))
    i = np.argwhere(indexes>0)
    for x in range(maxi):
        coeffA[x,x] = 4
        p,q = i[x]
        if indexes[p-1,q] != 0: 
            upper = int(indexes[p-1,q]-1)
            coeffA[x][upper] = -1
        if indexes[p,q-1] != 0: 
            left = int(indexes[p,q-1] -1)
            coeffA[x][left] = -1
        if indexes[p+1,q] != 0: 
            lower = int(indexes[p+1,q] -1)
            coeffA[x][lower] = -1

        if indexes[p,q+1] != 0:
            right = int(indexes[p,q+1] -1)
            coeffA[x][right] = -1
    return coeffA