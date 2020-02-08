'''
  File name: nonMaxSup.py
  Author:
  Date created:
'''

'''
  File clarification:
    Find local maximum edge pixel using NMS along the line of the gradient
    - Input Mag: H x W matrix represents the magnitude of derivatives
    - Input Ori: H x W matrix represents the orientation of derivatives
    - Output M: H x W binary matrix represents the edge map after non-maximum suppression
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

def nonMaxSup(Mag, Ori):
    P, Q = Mag.shape
    
    Arr_x = np.linspace(0,Q-1,Q)
    Arr_y = np.linspace(0,P-1,P)
    Vx, Vy = np.meshgrid(Arr_x,Arr_y)
    
    dx = np.add(np.cos(Ori),Vx)
    dy = np.add(np.sin(Ori),Vy)
    ddx = np.subtract(Vx,np.cos(Ori))
    ddy = np.subtract(Vy,np.sin(Ori))
    
    Mag_d = interp2(Mag,dx,dy)
    Mag_dd = interp2(Mag,ddx,ddy)

    M1 = np.logical_and(np.greater(Mag,Mag_d),np.greater(Mag,Mag_dd))
    M2 = np.logical_and(np.greater(Mag_d,Mag),np.greater(Mag_d,Mag_dd))
    M3 = np.logical_and(np.greater(Mag_dd,Mag),np.greater(Mag_dd,Mag_d))
    M = np.logical_or(M1,M2,M3)
    
    
    return(M)
