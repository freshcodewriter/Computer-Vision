import numpy as np
from sklearn.neighbors import NearestNeighbors

# initialize parameter
def feat_match(descs1,descs2):
    N = descs1.shape[0]
    descriptor_size = 64
    match = np.zeros(N).astype(int)
    match.fill(-1)
    threshold = 0.7

    nbrs_farward = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(descs2)
    distances_farward, indices_farward = nbrs_farward.kneighbors(descs1)

    # bi-directional check
#     nbrs_backward = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(descs1)
#     distances_backward, indices_backward = nbrs_backward.kneighbors(descs2)

    for i in range(N):
        ratio_farward = distances_farward[i,0] / distances_farward[i,1]
#         ratio_backward = distances_backward[i,0] / distances_backward[i,1]
        if ratio_farward < threshold:
#         if ratio_farward < threshold and ratio_backward < threshold:
            match[i] = indices_farward[i,0]
        
    return match

