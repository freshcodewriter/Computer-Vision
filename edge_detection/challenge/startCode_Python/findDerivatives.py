'''
  File name: findDerivatives.py
  Author: Jiaxiao Cai
  Date created: Sep-16-2019 
'''

'''
  File clarification:
    Compute gradient information of the input grayscale image
    - Input I_gray: H x W matrix as image
    - Output Mag: H x W matrix represents the magnitude of derivatives
    - Output Magx: H x W matrix represents the magnitude of derivatives along x-axis
    - Output Magy: H x W matrix represents the magnitude of derivatives along y-axis
    - Output Ori: H x W matrix represents the orientation of derivatives
'''
import numpy as np
from scipy import signal
from scipy import interpolate
from utils import GaussianPDF_2D
from utils import rgb2gray
from interp import interp2
from PIL import Image
from utils import visDerivatives
import matplotlib.pyplot as plt 


def findDerivatives(I_gray): 
    G = GaussianPDF_2D(0, 1, 5, 5)
    dx, dy = np.gradient(G, axis = (1,0))
    Magx = signal.convolve2d(I_gray,dx,'same')
    Magy = signal.convolve2d(I_gray,dy,'same')
    Mag = np.sqrt(Magx*Magx+Magy*Magy)
    Ori = np.arctan2(Magy, Magx)
    return Mag, Magx, Magy, Ori







