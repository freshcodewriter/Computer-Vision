'''
  File name: rmVerSeam.py
  Author:
  Date created:
'''

'''
  File clarification:
    Removes vertical seams. You should identify the pixel from My from which 
    you should begin backtracking in order to identify pixels for removal, and 
    remove those pixels from the input image. 
    
    - INPUT I: n × m × 3 matrix representing the input image.
    - INPUT Mx: n × m matrix representing the cumulative minimum energy map along vertical direction.
    - INPUT Tbx: n × m matrix representing the backtrack table along vertical direction.
    - OUTPUT Ix: n × (m - 1) × 3 matrix representing the image with the row removed.
    - OUTPUT E: the cost of seam removal.
'''
import numpy as np

def rmVerSeam(I, Mx, Tbx):
    height = Mx.shape[0]
    width = Mx.shape[1]
    Ix = np.zeros((height,width-1,3),dtype = np.uint8)
    idx = np.argmin(Mx[height-1:])
    Ex = Mx[height-1,idx]
    for i in reversed(range(0,height)):
        left = I[i:i+1,:int(idx),:]
        right = I[i:i+1,int(idx)+1:,:]
        Ix[i:i+1,:,:] = np.concatenate((left,right),axis=1)
        idx = Tbx[i,int(idx)]
    return Ix, Ex