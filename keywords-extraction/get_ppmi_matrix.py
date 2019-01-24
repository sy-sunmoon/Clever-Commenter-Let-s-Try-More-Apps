
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


def jaccard_index(sent1, sent2):
    a = set(sent1.split(' '))
    b = set(sent2.split(' '))
    c = a.intersection(b)
    d = a.union(b)
    return float(len(c)) / len(d)

def get_score(form,x,y):
    score = jaccard_index( str(form[x][0]) , str(form[y][0]) )
    score += ( str(form[x][9]) == str(form[y][9]) )
    score += ( str(form[x][1])== str(form[y][1]) )
    score += 0.2*( str(form[x][6])== str(form[y][6]) )
    score += 0.2*( str(form[x][8])== str(form[y][8]) )
    if str(form[x][12]).split(' ')[0].isdigit() and str(form[y][12]).split(' ')[0].isdigit():
        score += 0.2*(str(form[x][12])[0]==str(form[y][12])[0])
    return score

appsum = len(form_app)
name_to_no = {}
no_to_name = {}
orl_power_matrix = np.zeros((appsum,appsum))
orl_matrix = form_app.as_matrix()
orl_matrix = np.array(orl_matrix)

for i in range(appsum):
    name_to_no[ str(orl_matrix[0]) ] = i
    no_to_name[i] = name_to_no[ str(orl_matrix[0]) ]
    print(i)
    for j in range(i,appsum):
        if i!=j:
            orl_power_matrix[i][j] = get_score(orl_matrix,i,j)
            orl_power_matrix[j][i] = orl_power_matrix[i][j]
               


# In[4]:


def change_0_to_1(data_matrix,appsum):
    min_num=np.nanmin(data_matrix)
    max_num=np.nanmax(data_matrix)
    ans_matrix = [ (x-min_num)/(max_num-min_num) for x in data_matrix]
    
    
    return ans_matrix

normalization_matrix = change_0_to_1(orl_power_matrix,appsum)


# In[5]:


def get_qianzhuihe(data_matrix,length):
    
    qianzhui_probability=np.zeros((length,length))
    
    for i in range(length):
        for j in range(length):
            if i!=j:
                qianzhui_probability[i][j]=data_matrix[i][j] 
        for j in range(1,length):
                qianzhui_probability[i][j]=qianzhui_probability[i][j]+qianzhui_probability[i][j-1]
        for j in range(0,length):
                qianzhui_probability[i][j]=qianzhui_probability[i][j]/qianzhui_probability[i][length-1]
    
    return qianzhui_probability

qianzhui_matrix = get_qianzhuihe(normalization_matrix,appsum)


# In[6]:


def random_walk(repeat_time,walk_length,length,qianzhui_probability):
    
    ans_matrix=np.zeros((length,length))
    begin_pot=0
    end_pot=0
    last_pot=-1

    for app in range(length):
        print(app)
        for repeat in range(repeat_time):
            last_pot=-1
            begin_pot=app
            for walk_now in range(walk_length):
                random_seed=random.uniform(0, 1)
                for i in range(length):
                    if(random_seed<qianzhui_probability[begin_pot][i] ):
                        #print(random_seed,qianzhui_probability[i],bian_no[i])
                        #print(begin_pot,random_seed,i,qianzhui_probability[begin_pot][i])
                        end_pot=i
                        break;
                #print(begin_pot,end_pot,random_seed)

                if last_pot==-1:
                        ans_matrix[int(begin_pot)][int(end_pot)] = ans_matrix[int(begin_pot)][int(end_pot)]+1
                        ans_matrix[int(end_pot)][int(begin_pot)] = ans_matrix[peo][int(end_pot)][int(begin_pot)]+1
                else:
                    ans_matrix[int(last_pot)][int(end_pot)] = ans_matrix[int(last_pot)][int(end_pot)]+1
                    ans_matrix[int(end_pot)][int(last_pot)] = ans_matrix[int(end_pot)][int(last_pot)]+1
                    ans_matrix[int(begin_pot)][int(last_pot)] = ans_matrix[int(begin_pot)][int(last_pot)]+1
                    ans_matrix[int(last_pot)][int(begin_pot)] = ans_matrix[int(last_pot)][int(begin_pot)]+1
                    ans_matrix[int(begin_pot)][int(end_pot)] = ans_matrix[int(begin_pot)][int(end_pot)]+1
                    ans_matrix[int(end_pot)][int(begin_pot)] = ans_matrix[int(end_pot)][int(begin_pot)]+1

                last_pot=int(begin_pot)
                 begin_pot=int(end_pot)
                    
                                       
    return ans_matrix


rm_ans = random_walk(50,15,appsum,qianzhui_matrix)                            


# In[7]:


def change_the_matrix(random_walk_matrix,length):
    ans_matrix=np.zeros((length,length))

    p_hang=np.zeros(length)
    p_lie=np.zeros(length)
    p_sum=0.0

    for i in range(length):
        for j in range(length):
            p_hang[i]=p_hang[i]+random_walk_matrix[i][j]
            p_lie[j]=p_lie[j]+random_walk_matrix[i][j]
            p_sum=p_sum+random_walk_matrix[i][j]
    for i in range(length):
        for j in range(length):
            if random_walk_matrix[i][j]!=0.0:
                ans_matrix[i][j]=max( math.log( (random_walk_matrix[i][j]*p_sum)/( p_hang[i]*p_lie[j] ) )  ,  0 )
                #print(math.log( (random_walk_matrix[peo][i][j]*p_sum)/( p_hang[i]*p_lie[j] )))
    
    return ans_matrix

pmi_matrix = change_the_matrix(rm_ans,appsum)


# In[8]:


np.savetxt('/home/suny/google-winter-camp/pmi_matrix',pmi_matrix,delimiter=' ')

