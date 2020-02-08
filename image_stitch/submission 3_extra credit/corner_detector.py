'''
  File name: corner_detector.py
  Author:
  Date created:
'''
import numpy as np
from anms import anms
from scipy import signal
'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
from skimage.feature import corner_harris
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg

def corner_detector(img):
  # Your Code Here
  cimg = corner_harris(img, method='k', k=0.01, eps=1e-06, sigma=3)
  return cimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

# img0 = np.array(Image.open("left.jpg"))
# img = mpimg.imread("left.jpg")
# img = rgb2gray(img0)
# corner = corner_detector(img)
# result = anms(corner,2000)
#
# fig, ax = plt.subplots()
# ax.imshow(img0)
# ax.plot(result[0], result[1], color='r', marker='o',
#         linestyle='None', markersize=1)
# plt.show()

# fig, axes = plt.subplots(1, 2)
# ax = axes.ravel()
# ax[0].imshow(img)
# ax[0].set_title('Original')

# ax[1].imshow(img)
# ax[1].autoscale(False)
# ax[1].plot(result[0], result[1], 'r.')
# ax[1].set_title('Peak local max')

# plt.show()