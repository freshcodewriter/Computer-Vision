%matplotlib qt
import numpy as np
import os
from PIL import Image
from scipy import misc
from click_correspondences import click_correspondences
from morph_tri import morph_tri
import matplotlib
import matplotlib.pyplot as plt

# test triangulation morphing
def test_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
  # generate morphed image
    morphed_ims = morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac)
    
    return True


# the main test code
def main():
    
    folder = 'images'

  # image1
    im_path = os.path.join(folder,"mia.jpg")
    im1 = np.array(Image.open(im_path).convert('RGB'))

  # image2
    im_path = os.path.join(folder,"bulldog.jpg")
    im2 = np.array(Image.open(im_path).convert('RGB'))
    
    im1_pts, im2_pts = click_correspondences(im1, im2)
    
    if(im1_pts.shape != im2_pts.shape):
        print("Please try again! The number of corresponding points for both pictures should be the same.")
        raise SystemExit
    im1_array = np.asarray(im1)
    im2_array = np.asarray(im2)
    warp_frac, dissolve_frac =  np.arange(0, 1, 0.01), np.arange(0, 1, 0.01) 

    if not test_tri(im1_array, im2_array, im1_pts, im2_pts, warp_frac, dissolve_frac):
        return
    
    print("All tests passed! ")
    return


if __name__ == "__main__":
    main()