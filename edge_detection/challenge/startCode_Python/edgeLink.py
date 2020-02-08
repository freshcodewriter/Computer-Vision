'''
  File name: edgeLink.py
  Author:
  Date created:
'''

'''
  File clarification:
    Use hysteresis to link edges based on high and low magnitude thresholds
    - Input M: H x W logical map after non-max suppression
    - Input Mag: H x W matrix represents the magnitude of gradient
    - Input Ori: H x W matrix represents the orientation of gradient
    - Output E: H x W binary matrix represents the final canny edge detection map
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

def edgeLink(M, Mag, Ori):
    mean = np.mean(Mag)
    std = np.std(Mag)
    maxi = Mag.max()
    mini = Mag.min()
    
#   after manually tune using test data, the regression of the highThreshold and lowThreshold with mean and standard deviation could be found as follows
    highThreshold = mean+8*std
    lowThreshold = mean
    
    P, Q = Mag.shape 
    E = M
    Arr_x = np.linspace(0,Q-1,Q)
    Arr_y = np.linspace(0,P-1,P)
    Vx, Vy = np.meshgrid(Arr_x,Arr_y)

    
    dx = np.add(np.cos(Ori+np.pi/2),Vx)
    dy = np.add(np.sin(Ori+np.pi/2),Vy)
    ddx = np.subtract(Vx,np.cos(Ori+np.pi/2))
    ddy = np.subtract(Vy,np.sin(Ori+np.pi/2))

    Mag_d = interp2(Mag,dx,dy)
    Mag_dd = interp2(Mag,ddx,ddy)
    
    zero = np.bool(False)
    strong = np.bool(True)
    
    strong_i, strong_j = np.where(Mag >= highThreshold)
    zeros_i, zeros_j = np.where(Mag < lowThreshold)
    weak_i, weak_j = np.where((Mag < highThreshold) & (Mag >= lowThreshold) & ((Mag_d >= highThreshold)|(Mag_dd >= highThreshold)))
    
    E[strong_i, strong_j] = strong
    E[zeros_i, zeros_j] = zero
    E[weak_i,weak_j] = strong
    
    return (E)

