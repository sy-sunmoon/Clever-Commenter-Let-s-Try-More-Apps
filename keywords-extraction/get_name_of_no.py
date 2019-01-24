
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys
import random
import math


# In[2]:


form_app = pd.read_csv("/home/suny/google-winter-camp/orl_data/googleplaystore.csv")
form_review = pd.read_csv("/home/suny/google-winter-camp/orl_data/googleplaystore_user_reviews.csv")


# In[3]:


address = '/home/suny/google-winter-camp/no_to_name'
f = open(address, "w")

appsum = len(form_app)
orl_power_matrix = np.zeros((appsum,appsum))
orl_matrix = form_app.as_matrix()
orl_matrix = np.array(orl_matrix)

for i in range(appsum):
    print(i,end='\t',file = f)
    print(str(orl_matrix[i][0]),file = f)

f.close()

