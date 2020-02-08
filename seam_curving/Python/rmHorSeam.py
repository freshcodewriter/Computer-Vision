'''
  File name: rmHorSeam.py
  Author:
  Date created:
'''

'''
  File clarification:
    Removes horizontal seams. You should identify the pixel from My from which 
    you should begin backtracking in order to identify pixels for removal, and 
    remove those pixels from the input image. 
    
    - INPUT I: n × m × 3 matrix representing the input image.
    - INPUT My: n × m matrix representing the cumulative minimum energy map along horizontal direction.
    - INPUT Tby: n × m matrix representing the backtrack table along horizontal direction.
    - OUTPUT Iy: (n − 1) × m × 3 matrix representing the image with the row removed.
    - OUTPUT E: the cost of seam removal.
'''
import numpy as np

def rmHorSeam(I, My, Tby):
    height = My.shape[0]
    width = My.shape[1]
    Iy = np.zeros((height-1,width,3),dtype = np.uint8)
    idx = np.argmin(My[:,width-1])
    Ey = My[idx,width-1]
    for i in reversed(range(0,width)):
        up = I[:int(idx),i:i+1,:]
        down = I[int(idx)+1:,i:i+1,:]
        Iy[:,i:i+1,:] = np.concatenate((up,down),axis=0)
        idx = Tby[int(idx),i]
    return Iy, Ey