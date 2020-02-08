'''
  File name: getSolutionVect.py
  Author:
  Date created:
'''
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from numpy.linalg import inv
from scipy import signal



laplacian = np.asarray([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

def getSolutionVect(indexes, source, target, offsetX, offsetY):
    lapl_convolve =  signal.convolve2d(source,laplacian,'same')
    i = np.argwhere(indexes>0)
    maxi = int(np.amax(indexes))
    SolVectorb = np.zeros(maxi)
    for x in range(maxi):
        p,q = i[x]
        SolVectorb[x] = lapl_convolve[p-offsetX,q-offsetY]
        if indexes[p-1,q] == 0:
            SolVectorb[x] += target[p-1,q]
        if indexes[p+1,q] == 0:
            SolVectorb[x] += target[p+1,q]
        if indexes[p,q-1] == 0:
            SolVectorb[x] += target[p,q-1]
        if indexes[p,q+1] == 0:
            SolVectorb[x] += target[p,q+1]
    return SolVectorb