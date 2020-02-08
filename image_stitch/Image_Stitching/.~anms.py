'''
  File name: anms.py
  Author:
  Date created:
'''
import numpy as np
from skimage.feature import peak_local_max
from queue import PriorityQueue
'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''

# def anms(cimg, max_pts):
#   # Your Code Here
#   result = peak_local_max(cimg, num_peaks = max_pts)
#   x = result[:,1]
#   y = result[:,0]
#   rmax = 1
#   return x, y, rmax

def anms(cimg, max_pts):
  # Your Code Here
  H = cimg.shape[0]
  W = cimg.shape[1]
  ###########################################################
  ## comment this section out after first run to save time ##
  ###########################################################
  q = PriorityQueue()
  print(H,W)
  for i in range(H):
    for j in range(W):
      q.put((-cimg[i,j],[i,j]))
  points = []
  for i in range(2000):
    points+=[q.get()[1]]
  np.save('out', points)
  ###########################################################
  points=np.load('out.npy')
  print(points.shape, len(points))

  q = PriorityQueue()
  i=0
  for p1 in points:
    d=100000
    value = cimg[p1[0],p1[1]]
    for p2 in points:
      v = cimg[p2[0],p2[1]]
      if v>value and v<=value*1.2:
        d=min(d,((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**(1/2))
    q.put((-d,i,p1))
    i+=1
  x=[]
  y=[]
  rmax=0
  for i in range(max_pts):
    result = q.get()
    x+=[result[2][1]]
    y+=[result[2][0]]
    rmax=-result[0]
  return x, y, rmax