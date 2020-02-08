
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.neighbors import NearestNeighbors

# initialize parameter
def crossMatch(descs1,descs2):
    descs1 = descs1.T
    descs2 = descs2.T
    N = descs1.shape[0]
    descriptor_size = 64
    match = np.zeros(N).astype(int)
    match.fill(-1)

    nbrs_farward = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(descs2)
    distances_farward, indices_farward = nbrs_farward.kneighbors(descs1)

    # bi-directional check
    nbrs_backward = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(descs1)
    distances_backward, indices_backward = nbrs_backward.kneighbors(descs2)

    print(indices_farward[0])
    print(indices_backward[23])
    for i in range(N):
        tmp = indices_farward[i] 
        if indices_backward[tmp] == i:
            match[i] = indices_farward[i]
    return match

