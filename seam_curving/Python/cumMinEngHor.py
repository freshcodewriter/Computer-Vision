'''
  File name: cumMinEngHor.py
  Author:
  Date created:
'''

'''
  File clarification:
    Computes the cumulative minimum energy over the horizontal seam directions.
    
    - INPUT e: n × m matrix representing the energy map.
    - OUTPUT My: n × m matrix representing the cumulative minimum energy map along horizontal direction.
    - OUTPUT Tby: n × m matrix representing the backtrack table along horizontal direction.
'''
import numpy as np
from cumMinEngVer import cumMinEngVer

def cumMinEngHor(e):
    e = e.T
    My, Tby = cumMinEngVer(e)
    My, Tby = My.T, Tby.T
    return My, Tby