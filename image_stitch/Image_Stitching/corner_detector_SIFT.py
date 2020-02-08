
# coding: utf-8

# In[ ]:


def corner_detector_SIFT(img):
    kp = sift.detect(img,None)
    cimg = np.asarray([[p.pt[0], p.pt[1]] for p in kp])
    return cimg

