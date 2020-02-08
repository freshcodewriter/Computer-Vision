%matplotlib inline
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from cumMinEngHor import cumMinEngHor
from cumMinEngVer import cumMinEngVer
from rmHorSeam import rmHorSeam
from rmVerSeam import rmVerSeam
from genEngMap import genEngMap
from carv import carv
import time

image = Image.open('./image/f2.jpg')
image_array = np.asarray(image)
e = genEngMap(image_array)
Ic,T = carv(image_array,50,30,True)
timestr = time.strftime("%Y%m%d-%H%M%S")
plt.imsave('./result_img/{}.jpg'.format(timestr), Ic)
plt.imshow(Ic)