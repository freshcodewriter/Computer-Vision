
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import time
from cumMinEngHor import cumMinEngHor
from cumMinEngVer import cumMinEngVer
from rmHorSeam import rmHorSeam
from rmVerSeam import rmVerSeam
from genEngMap import genEngMap
from animation import animate

def carv(I, nr, nc,animation = True):
    
    T = np.zeros((nr+1,nc+1))
    dp = np.empty((nr+1,nc+1),dtype=np.ndarray)
    direction = np.zeros((nr+1,nc+1),dtype = str)
    dp[0,0] = I
    
    for i in range(nr):
        e = genEngMap(dp[i,0])
        My, Tby = cumMinEngHor(e)
        Iy, Ey = rmHorSeam(dp[i,0],My,Tby)
        dp[i+1,0] = Iy
        T[i+1,0] = Ey
        direction[i+1,0] = 'D'
        print('Step {} of {}'.format(i+1,nr))
    
    for i in range(nc):
        e = genEngMap(dp[0,i])
        Mx, Tbx = cumMinEngVer(e)
        Ix, Ex = rmVerSeam(dp[0,i],Mx,Tbx)
        dp[0,i+1] = Ix
        T[0,i+1] = Ex
        direction[0,i+1] = 'R'
        print('Step {} of {}'.format(i+1,nc))
        
    cnt = 0
    for i in range(nr):
        for j in range(nc):
            e_r = genEngMap(dp[i,j+1])
            My, Tby = cumMinEngHor(e_r)
            Iy, Ey = rmHorSeam(dp[i,j+1],My,Tby)
            Cost_r = T[i,j+1] + Ey
            
            e_c = genEngMap(dp[i+1,j])
            Mx, Tbx = cumMinEngVer(e_c)
            Ix, Ex = rmVerSeam(dp[i+1,j],Mx,Tbx)
            Cost_c = T[i+1,j] + Ex
            
            if Cost_c < Cost_r:
                T[i+1,j+1] = Cost_c
                dp[i+1,j+1] = Ix
                direction[i+1,j+1] = 'D'
            else:
                T[i+1,j+1] = Cost_r
                dp[i+1,j+1] = Iy
                direction[i+1,j+1] = 'R'
                
            cnt += 1
            print('Step {} of {}'.format(cnt,nr*nc))
    
    if animation == True:
        print('Generate Gif...')
        frame_list = animate(dp, direction, nr, nc)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        imageio.mimsave('./result_gif/{}.gif'.format(timestr), frame_list)
        print('Process Completed')
        
    Ic = dp[nr,nc]
    return Ic,T

