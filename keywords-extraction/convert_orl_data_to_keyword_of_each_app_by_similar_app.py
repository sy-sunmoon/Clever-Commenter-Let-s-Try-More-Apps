
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys
import jieba
import jieba.analyse
from rake_nltk import Metric, Rake
import nltk
import string

from nltk.corpus import stopwords
from nltk import word_tokenize
stop = set(stopwords.words('english'))


# In[2]:


form_app = pd.read_csv("/home/suny/google-winter-camp/orl_data/googleplaystore.csv")
form_review = pd.read_csv("/home/suny/google-winter-camp/orl_data/googleplaystore_user_reviews.csv")


# In[3]:


linenum = 0
similar_app = {}
for line in open('/home/suny/google-winter-camp/similar_app_no'):
    line = line.strip().split('	')
    linenum = linenum + 1
    line_no = str(line[1]).strip().split(' ')
    no_list=[]
    for i in range(len(line_no)):
        no_list.append(int(line_no[i]))
    similar_app[float(line[0])] =  no_list


# In[4]:


len_app = len(form_app)
len_review = len(form_review)


app_matrix = form_app.as_matrix()
app_matrix = np.array(app_matrix)

review_matrix = form_review.as_matrix()
review_matrix = np.array(review_matrix)


# In[5]:


#获得app对应的类别dict，和类别list

def get_app_to_category_and_category_list(form):

    dict_app_to_category = {}
    namelist=[]

    len_form = len(form)
    for i in range(len_form):    
        if form.iloc[i]['App'] not in dict_app_to_category:
            dict_app_to_category[form.iloc[i]['App']] = form.iloc[i]['Category']
        if form.iloc[i]['Category'] not in namelist:
            namelist.append(form.iloc[i]['Category'])
            
    return dict_app_to_category , namelist
app_name_to_category , category_name_list = get_app_to_category_and_category_list(form_app)


# In[6]:


#获取每一个APP相关的app的所有的句子

def get_each_sentence_by_app(form1,form2,no_dict,len1,len2):

    sentence_by_app = {}

    for i in range(len1):
        print(i)
        
        index = no_dict[i]
        
        name_list = form1[index][0]
        
        if str(form1[i][0]) not in sentence_by_app:
            sentence_by_app[str(form1[i][0])] = []
        
        for j in range(len2):
   
            if  str(form2[j][0]) in name_list:
                sentence_by_app[str(form1[i][0])].append(str(form2[j][1]))
    
    return sentence_by_app

review_by_app = get_each_sentence_by_app(app_matrix,review_matrix,similar_app,app_matrix.shape[0],review_matrix.shape[0])


# In[7]:


#建立情感极性词表
def init_emotion_list(threshold):
    emotion_list = []
    for line in open("/home/suny/google-winter-camp/emotion_data/vader_lexicon.txt"):
        line = line.strip().split('	')
        if( abs(float(line[1])) > threshold ):
            emotion_list.append( str(line[0]) )
        
    return emotion_list
        

#获得每个APP对应的关键词，关键词排除了情感极性词
def get_each_app_keyword(dict_sentence,topcnt,threshold): 
    
    
    emotion_word_list = init_emotion_list(threshold)
    
    dict_keyword_not_emotion={}
    dict_keyword_just_emotion={}
    
    for key in dict_sentence:
        
        sentence_sum = ''
        
        for i in range(len(dict_sentence[key])):
            
            each_review = dict_sentence[key][i]
            token_text = word_tokenize(each_review)
            new_review=""
            for single_words in token_text:
                if single_words not in stop and single_words not in string.punctuation:
                    single_words = single_words.lower()
                    new_review += single_words+" "
            sentence_sum += new_review
            
        #rake-nltk  
        #r = Rake(max_length=1)
        #r.extract_keywords_from_text(sentence_sum)
        #this_category_keyword = r.get_ranked_phrases()
           
        #结巴分词
        this_app_keyword = jieba.analyse.extract_tags(sentence_sum.encode('utf-8'), topK=topcnt, withWeight=False, allowPOS=())
   

        this_app_keyword_after_wash_emotion_word=[]
        this_app_keyword_after_just_emotion_word=[]
        for word in this_category_keyword:
            word.encode('utf-8')
            if word not in emotion_word_list:
                this_app_keyword_after_wash_emotion_word.append(word)
            else:
                this_app_keyword_after_just_emotion_word.append(word)
        
        dict_keyword_not_emotion[key] = this_app_keyword_after_wash_emotion_word
        dict_keyword_just_emotion[key] = this_app_keyword_after_just_emotion_word
        
    return dict_keyword_not_emotion , dict_keyword_just_emotion

app_and_keyword_no_emotion , app_and_keyword_just_emotion = get_each_app_keyword(review_by_app,100,1.5)            


# In[8]:


def print_to_file(file_address):
    f = open(file_address, "w")
    for key in category_and_keyword_no_emotion:
        
        print(key, end="\t", file = f)
        
        print(len(category_and_keyword_no_emotion[key]), end="\t", file = f)
        
        for i in range(len(category_and_keyword_no_emotion[key])):
            print(category_and_keyword_no_emotion[key][i], end=" ", file = f)
        print(end="\t",file = f)
        
        print(len(category_and_keyword_just_emotion[key]), end="\t", file = f)
        
        for i in range(len(category_and_keyword_just_emotion[key])):
            print(category_and_keyword_just_emotion[key][i], end=" ", file = f)
        print(end="\t",file = f)
        
        print(file = f)
        
    f.close()
print_to_file("/home/suny/google-winter-camp/keyword/similar_app_keyword_of_each_app_threshold_1.5_top_100_jieba")

