'''
  File name: cumMinEngVer.py
  Author:
  Date created:
'''

'''
  File clarification:
    Computes the cumulative minimum energy over the vertical seam directions.
    
    - INPUT e: n × m matrix representing the energy map.
    - OUTPUT Mx: n × m matrix representing the cumulative minimum energy map along vertical direction.
    - OUTPUT Tbx: n × m matrix representing the backtrack table along vertical direction.
'''
import numpy as np

def cumMinEngVer(e):
    r = e.shape[0]
    c = e.shape[1]
    
    Mx = e.copy()
    Tbx = np.zeros_like(Mx, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                idx = np.argmin(Mx[i - 1, j:j + 2])
                Tbx[i, j] = idx + j
                min_energy = Mx[i - 1, idx + j]
            else:
                idx = np.argmin(Mx[i - 1, j - 1:j + 2])
                Tbx[i, j] = idx + j - 1
                min_energy = Mx[i - 1, idx + j - 1]

            Mx[i, j] += min_energy

    return Mx, Tbx