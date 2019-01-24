
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys
import random
import math
import heapq


# In[2]:


pmi_matrix=np.loadtxt('/home/suny/google-winter-camp/pmi_matrix',delimiter=' ')

no_to_name = {}
linenum = 0
for line in open("/home/suny/google-winter-camp/no_to_name"):
    line = line.strip().split('	')
    linenum = linenum + 1
    if str(line[0]) not in no_to_name:
        no_to_name[ str(line[0]) ] = str(line[1])


# In[3]:


k = 10
file_address_name = '/home/suny/google-winter-camp/similar_app_name'
file_address_no = '/home/suny/google-winter-camp/similar_app_no'
f_name = open(file_address_name, "w")
f_no = open(file_address_no, "w")
for i in range(linenum):
    big_index = map(pmi_matrix[i].tolist().index, heapq.nlargest(k, pmi_matrix[i].tolist()))
    big_index = list(big_index)
    print(no_to_name[str(i)] , end='\t' , file = f_name)
    print(i , end='\t' , file = f_no)
    for j in range(len(big_index)):
        print(no_to_name[str(big_index[j])] , end=' ' , file = f_name)
        print(big_index[j] , end=' ' , file = f_no)
    print( file = f_name )
    print( file = f_no )

