'''
  File name: getIndexes.py
  Author:
  Date created:
'''
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from numpy.linalg import inv
from scipy import signal


def getIndexes(mask, targetH, targetW, offsetX, offsetY):
    indexes = np.zeros((targetH, targetW))
    source_size = mask.shape
    indexes[offsetX:offsetX+source_size[0],offsetY:offsetY+source_size[1]] = mask
    nonzero_cnt = np.count_nonzero(mask)
    i,j = np.where(indexes!=0)
    indexes[i,j] = np.arange(nonzero_cnt)+1 
    return indexes