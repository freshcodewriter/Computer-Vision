
# coding: utf-8

# In[ ]:
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

LOW_THRESH = 0.5
HIGH_THRESH = 0

def testing_anms(corner_matrix, max_pts):
  h, w = corner_matrix.shape[0:2]
  inf = float("inf")

  ''' Preparing corner map '''

  # Low thresholding based on mean value of the corner matrix
  mean_value = np.mean(corner_matrix)
  threshold = LOW_THRESH * mean_value
  corner_matrix[np.where(corner_matrix < threshold)] = 0.0

  # From thresholded corner map, find the local maxima
  corner_coords = peak_local_max(corner_matrix, min_distance=1, exclude_border=True)
  num_corners = corner_coords.shape[0]


  ''' Prepare the M*M distance matrix '''

  # a[i,j] == distance between ith and jth corners
  distances = np.zeros((num_corners, num_corners)).astype(float)
  idx = np.arange(0, num_corners)

  # Iterative version
  for i in idx:
    this_coords = corner_coords[i]

    for j in idx:
      other_coords = corner_coords[j]

      # Compute distance
      distances[i,j] = np.sqrt((this_coords[0] - other_coords[0])**2 + (this_coords[1] - other_coords[1])**2)

  # Set self distances to infinity
  distances[idx, idx] = inf

  # For each corner...
  for i in idx:
    this_coords = corner_coords[i]
    this_corner = corner_matrix[this_coords[0], this_coords[1]]

    # Check against every other corner...
    other_corners = corner_matrix[corner_coords[:,0], corner_coords[:,1]]
    j = np.where(other_corners < HIGH_THRESH * this_corner)
    distances[i, j] = inf


  ''' Find the optimal points from the distance matrix '''

  # Get the argmin from each row
  closest_maxima = np.argmin(distances, axis=1)
  rmax = np.max(np.min(distances, axis=1))

  # Best corners are the minima of each each row
  best_corners = np.unique(corner_coords[closest_maxima], axis=0)


  ''' If we get more than max_pts, just keep the best of those '''
  if (best_corners.shape[0] > max_pts):
    corner_values = corner_matrix[best_corners[:,0], best_corners[:,1]]

    # Get the smallest possible point we're going to keep
    all_chosen = np.flipud(np.sort(corner_values))
    top_threshold = all_chosen[max_pts]

    # Get rid of indices corresponding to smaller points
    best_corners = best_corners[np.where(corner_values > top_threshold)]


  ''' Return '''
  x = best_corners[:,1]
  y = best_corners[:,0]
  return x, y, rmax

