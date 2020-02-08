
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import dlib
from imutils import face_utils
from helper import Blend_faces

# Define detector, predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Read replacement video
cap2 = cv2.VideoCapture('Dataset/Myvideo/caci.mp4')
if (cap2.isOpened() == False): 
    print("Please import replacement video")
ret2, image2 = cap2.read()
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
rects_cap2 = detector(gray2, 1)
for (i, rect) in enumerate(rects_cap2):
    shape_cap2 = predictor(gray2, rect)
    shape_cap2 = face_utils.shape_to_np(shape_cap2)

# Read source video
cap1 = cv2.VideoCapture('Dataset/Medium/LucianoRosso3.mp4')
if (cap1.isOpened() == False): 
    print("Please import source video")
    
ret1, image1 = cap1.read()
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
rects_cap1 = detector(gray1, 1)
for (i, rect) in enumerate(rects_cap1):
    shape_cap1 = predictor(gray1, rect)
    shape_cap1 = face_utils.shape_to_np(shape_cap1)
    # Detect one face at a time 
    if(i == 1):
        break
        
# Define output video
cap1_width = int(cap1.get(3))
cap1_height = int(cap1.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (cap1_width,cap1_height))

# Optical flow
lk_params = dict(winSize  = (15,15),maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
landmarks = shape_cap1.reshape([68,1, 2])
prev_point = landmarks.astype(np.float32)
prev_weighted = prev_point

# Use facial landmarks as feature points to perform optical flow
while(1):
    new_weighted = np.zeros([68,1,2])
    ret, image = cap1.read()
    flag = 0
    
    if ret == 1:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prev_point = prev_point.astype(np.float32)
        
        # compute the new point
        new_point, st, err = cv2.calcOpticalFlowPyrLK(gray1, image_gray, prev_point, None, **lk_params)
        rects_new = detector(image_gray, 1)
        for (i, rect) in enumerate(rects_new):
            new_shape = predictor(image_gray, rect)
            new_shape = face_utils.shape_to_np(new_shape)
            if (i == 1):
                break
            
        landmarks_new = new_shape.reshape(68, 1, 2)
        
        if new_shape.shape[0] == 68:
            flag = 1
        
        #for each point, compute a runnin average with higher weight on KLT
        for point in range(68):
            if (flag == 1 and st[point] == 1):
                new_weighted[point, :, :] = 0.2*landmarks_new[point, :, :] + 0.8*new_point[point, :, :]
            elif (flag == 1 and st[point] == 0):
                new_weighted[point, :, :] = 0.8*prev_weighted[point, :, :] + 0.2*landmarks_new[point, :, :]
            elif (flag == 0 and st[pt_num] == 1):
                new_weighted[point, :, :] = 0.2*prev_weighted[point, :, :] + 0.8*new_point[point, :, :]
            else:
                new_weighted[point, :, :] = prev_weighted[point, :, :]
                
        new_weighted = new_weighted.reshape([68,2]).astype(int)
        new_weighted_copy = np.copy(new_weighted)
        shape_cap2_copy = np.copy(shape_cap2)
        image_copy = np.copy(image)
        image2_copy = np.copy(image2)
        
        # blend faces
        faces = Blend_faces(new_weighted_copy, shape_cap2_copy, image_copy, image2_copy)             

        out.write(faces)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        # Now update the previous frame and previous points
        gray1 = image_gray.copy()
        prev_point = new_weighted.reshape(-1,1,2)
        
    else:
        break
    
cv2.destroyAllWindows()
cap1.release()
out.release()

