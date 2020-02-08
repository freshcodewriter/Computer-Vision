
# coding: utf-8

# In[ ]:


import numpy as np

def animate(dp, direction, nr, nc):
    r = nr
    c = nc
    frames = np.zeros((nr+nc+1,),dtype=np.ndarray)
    frames[0] = dp[0,0]
    shape = np.shape(frames[0])
    
    while(r > 0 or c > 0):
        r = nr
        c = nc
        frames = np.zeros((nr+nc+1,),dtype=np.ndarray)
        frames[0] = dp[0,0]
        shape = np.shape(frames[0])
        frame_list = []
        while(r > 0 or c > 0):
            if direction[r,c] == 'R':
                frames[r+c] = dp[r,c]
                c -= 1
            elif direction[r,c] == 'D':
                frames[r+c] = dp[r,c]
                r -= 1
            frame = np.pad(dp[r,c],((0,shape[0] - dp[r,c].shape[0]),(0,shape[1] - dp[r,c].shape[1]),(0,0)),'constant', constant_values=(0))
            frame_list.append(frame)
            
        frame_list.append(frames[0])
        frame_list = frame_list[::-1]
        return frame_list
        
        

