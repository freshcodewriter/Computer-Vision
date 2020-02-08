import numpy as np
import matplotlib.pyplot as plt


def seam(left, right, pts1, pts2):
	print('--- seam ---')
	# exit()
	h=left.shape[0]
	w1=left.shape[1]
	w2=right.shape[1]
	lefteng = rgb2gray(left)[:,pts1[0,0]:]
	righteng = rgb2gray(right)[:,:pts2[3,0]+1]
	# exit()
	Mx, Tbx = cumMinEngVer(lefteng, righteng)
	result = np.zeros((h,(w1+w2)-(w1-pts1[0,0]+1),3))

	E = min(Mx[h-1,:])
	y=np.where(Mx[h-1,:]==E)[0][0]
	remove = [(h-1,y+pts1[0,0])]
	w=(w1+w2)-(w1-pts1[0,0]+1)
	for i in range(h-2,-1,-1):
		y=y+Tbx[i+1,y]
		remove+=[(i,y+pts1[0,0])]

	for i in range(h):
		skip = 0
		for j in range(w):
			if (i,j) in remove:
				skip = 1
			for k in range(3):
				if skip == 1:
					assert j-pts1[0,0] >= 0
					result[i,j,k] = right[i,j-pts1[0,0],k]
				else:
					result[i,j,k] = left[i,j,k]
	return result.astype('uint8')

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def cumMinEngVer(lefteng, righteng):
  # Your Code Here
  e=abs(lefteng-righteng)
  n=e.shape[0]
  m=e.shape[1]
  Mx, Tbx = np.zeros((n,m)), np.zeros((n,m),dtype=int)
  Mx[0,:] = e[0,:]
  for i in range(1, n):
    for j in range(m):
      value = min(Mx[i-1,max(j-1,0)],Mx[i-1,j],Mx[i-1,min(j+1,m-1)])
      Mx[i,j]=e[i,j]+value
      if j-1>= 0 and Mx[i-1,j-1] == value:
        Tbx[i,j]=-1
      elif j+1 <= m-1 and Mx[i-1,j+1] == value:
        Tbx[i,j]=1
  return Mx, Tbx