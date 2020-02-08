
# coding: utf-8

# In[ ]:
import cv2
sift = cv2.xfeatures2d.SIFT_create()

def feat_desc_SIFT(img,keypoints):
    kp, des= sift.compute(img,keypoints)
#     for i in range(0, des.shape[0]):
#         amax = np.amax(des[i])
#         amin = np.amin(des[i])
#         des[i] = (des[i]-amin) / (amax-amin)
    return des

