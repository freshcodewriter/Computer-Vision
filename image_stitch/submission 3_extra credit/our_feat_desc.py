'''
  File name: feat_desc.py
  Author:
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature, 
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40 
    window to have a nice big blurred descriptor. 
    - Input image: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
Mia is the best in the world and John is annoying 
'''

import numpy as np

DESC_SIZE = 64

def our_feat_desc(image, x, y):
  h, w = image.shape[0:2]
  num_corners = x.shape[0]
  corners = np.array((y, x))

  # Offsets
  offsets_row = np.tile(np.arange(-17.5, 20, 5), (8,1)).transpose()
  offsets_col = np.tile(np.arange(-17.5, 20, 5), (8,1))
  offsets = np.array((np.ravel(offsets_row), np.ravel(offsets_col)))

  ''' Loop: for each corner, generate descriptor '''
  descriptors = np.zeros( (DESC_SIZE, num_corners) )
  for n in np.arange(0, num_corners):

    # Get the current y,x pair
    corner = corners[:,n]

    # Generate indices of the pixels we will sample from
    sample_idx = offsets + corner.reshape((2,1))
    ## todo: deal with out of bounds (negative pixels and >h,w pixels)
    ## just replace with zeros in the sample???
    sample_idx = np.round(sample_idx).astype(int)
    sample_idx[0, sample_idx[0] < 0] = 0
    sample_idx[1, sample_idx[1] < 0] = 0
    sample_idx[0, sample_idx[0] >= h] = h-1
    sample_idx[1, sample_idx[1] >= w] = w-1
    # Sample the pixels from the image, and reshape + standardize to form the descriptor
    samples = image[sample_idx[0], sample_idx[1]]
    descriptors[:,n] = (samples - np.mean(samples)) / np.std(samples)

  return descriptors
