from skimage.feature import corner_harris

def corner_detector(img):
  cimg = corner_harris(img, method='k', k=0.01, eps=1e-06, sigma=3)
  return cimg