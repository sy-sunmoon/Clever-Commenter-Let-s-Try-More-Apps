
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


app_to_category_with_repetition = form_app[['App','Category']]


# In[4]:


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


# In[5]:


app_name_to_category , category_name_list = get_app_to_category_and_category_list(app_to_category_with_repetition)


# In[6]:


#获取每一个类别的所有句子

def get_each_sentence_by_category(form,app_name_to_category,category_name_list):

    sentence_by_category = {}

    lenform = len(form)

    for i in range(lenform):

        app_name=form.iloc[i]['App']

        #有一些APP没有对应的分类，于是新建分类OTHERS
        if app_name not in app_name_to_category:
            app_name_to_category[app_name] = 'OTHERS'
        if app_name not in category_name_list:
            category_name_list.append('OTHERS')

        category_name = app_name_to_category[app_name]

        #如果评论是nan，那么不记录，直接过滤掉

        if pd.isnull(form.iloc[i]['Translated_Review'])== False:

            if category_name not in sentence_by_category:
                sentence_by_category[category_name] = []
                sentence_by_category[category_name].append(form.iloc[i]['Translated_Review'])
            else:
                sentence_by_category[category_name].append(form.iloc[i]['Translated_Review'])
                
    return sentence_by_category
    


# In[7]:


review_by_category = get_each_sentence_by_category(form_review,app_name_to_category,category_name_list)


# In[8]:


#建立情感极性词表
def init_emotion_list(threshold):
    emotion_list = []
    for line in open("/home/suny/google-winter-camp/emotion_data/vader_lexicon.txt"):
        line = line.strip().split('	')
        if( abs(float(line[1])) > threshold ):
            emotion_list.append( str(line[0]) )
        
    return emotion_list
        

#获得每个类别对应的关键词，关键词排除了情感极性词
def get_each_category_keyword(dict_sentence,topcnt,threshold): 
    
    
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
        this_category_keyword = jieba.analyse.extract_tags(sentence_sum.encode('utf-8'), topK=topcnt, withWeight=False, allowPOS=())
   

        this_category_keyword_after_wash_emotion_word=[]
        this_category_keyword_after_just_emotion_word=[]
        for word in this_category_keyword:
            word.encode('utf-8')
            if word not in emotion_word_list:
                this_category_keyword_after_wash_emotion_word.append(word)
            else:
                this_category_keyword_after_just_emotion_word.append(word)
        
        dict_keyword_not_emotion[key] = this_category_keyword_after_wash_emotion_word
        dict_keyword_just_emotion[key] = this_category_keyword_after_just_emotion_word
        
    return dict_keyword_not_emotion , dict_keyword_just_emotion
            


# In[9]:


category_and_keyword_no_emotion , category_and_keyword_just_emotion = get_each_category_keyword(review_by_category,100,1.5)


# In[10]:


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


# In[11]:


print_to_file("/home/suny/google-winter-camp/keyword/keyword_threshold_1.5_top_100_jieba")


# In[12]:


category_and_keyword_no_emotion

