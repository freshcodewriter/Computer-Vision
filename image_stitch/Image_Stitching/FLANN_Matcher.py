
# coding: utf-8

# In[ ]:
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris
from PIL import Image
import matplotlib.image as mpimg
from anms import anms
sift = cv2.xfeatures2d.SIFT_create()


# FLANN_Matcher 
def FLANN_Matcher(img_1,img_2,descs_1,descs_2,keypoints_1,keypoints_2,xy_1,xy_2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(descs_1,descs_2,k=2)
    
    match = np.zeros((descs_1.shape[0],1))
    match.fill(-1)
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    good_without_list = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.6*n.distance:
            matchesMask[i]=[1,0]
            good_without_list.append(m)
            
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)

    img3 = cv2.drawMatchesKnn(img_1,keypoints_1,img_2,keypoints_2,matches,None,**draw_params)

    plt.imshow(img3,),plt.show()
    plt.imsave('./output_2.jpg',img3)
    
    list_kp1 = [keypoints_1[mat.queryIdx].pt for mat in good_without_list] 
    list_kp2 = [keypoints_2[mat.trainIdx].pt for mat in good_without_list]

    for i in range(len(list_kp1)):
        col_kp1, row_kp1 = list_kp1[i]
        col_kp2, row_kp2 = list_kp2[i]
        idx = np.argwhere((xy_1[:,0] == col_kp1) & (xy_1[:,1]== row_kp1)).flatten()
        val = np.argwhere((xy_2[:,0] == col_kp2) & (xy_2[:,1]== row_kp2)).flatten()
        match[idx[0]] = val[0]
        
    return match

