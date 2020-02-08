# %matplotlib qt
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from pylab import ginput
from PIL import Image

def click_correspondences(im1, im2):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,10))
    plt.setp((ax1, ax2), xticks=[0, 100, 200, 300, 400, 500, 600], xticklabels=['0', '100', '200','300','400','500','600'],
            yticks=[0, 100, 200, 300, 400, 500, 600], yticklabels=['0', '100', '200','300','400','500','600'])

    ax1.imshow(im1, aspect='auto')
    ax1.set_title('Image 1')

    ax2.imshow(im2, aspect='auto')
    ax2.set_title('Image 2')

    plt.tight_layout()
    plt.show()
    
    print("Please click") 
    points = np.asarray(ginput(0,0))
    points_size = points.shape[0]
#     print (points)

    index_im1 = np.arange(0,points_size-1,2)
#     print(index_im1)
    index_im2 = np.arange(1,points_size,2)
#     print(index_im2)

    im1_pts = points[index_im1]
#     print(im1_pts)
    im2_pts = points[index_im2]
#     print(im2_pts)
    
    return im1_pts, im2_pts
