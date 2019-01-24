# -*- coding: utf-8 -*-  
"""  
 @version: python2.7 
 @author: luofuli 
 @time: 2019/1/22 10:01 
"""
import csv
from rake_nltk import Rake
from nltk import word_tokenize
import nltk
import numpy as np
import os
import random


def get_topic_to_review_data(src_path, dst_path):
    key_words = []

    count = 0
    with open(src_path) as f, open(dst_path, 'w') as fw:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for i, row in enumerate(f_csv):
            review = row[1].strip().decode('ascii', errors='ignore')
            senti = row[2]

            if review == u'nan':
                continue

            text = word_tokenize(review)
            text = [word.lower() for word in text]

            if len(text) < 5 or len(text) > 30:
                continue

            tagged_text = nltk.pos_tag(text)
            key_word = []
            for word, tag in tagged_text:
                if 'NN' in tag or 'VB' in tag or 'JJ' in tag or 'RB' in tag:  # JJ adjective, RB adverb
                    # if 'NN' in tag or 'VB' in tag:
                    if word not in key_word and len(word) > 1:
                        key_word.append(word)

            if len(key_word) == 0:
                continue

            k_word = len(text) / 5 + 1  # each 5 word has a key-word
            sampled = []
            if len(key_word) > k_word:
                k = len(key_word) / k_word + 1
                for j in range(k):
                    # tmp_key_word = np.random.choice(key_word, 3)
                    tmp_key_word = random.sample(key_word, k_word)
                    new_tmp_key_word = []
                    for w in text:
                        if w in tmp_key_word:
                            new_tmp_key_word.append(w)
                    sampled_key_word = ' '.join(new_tmp_key_word)

                    if sampled_key_word not in sampled:
                        sampled.append(sampled_key_word)
                        fw.write((' '.join(new_tmp_key_word) + '\t' + ' '.join(text) + '\n').encode('utf-8'))
                        count += 1
            else:
                key_words.append(key_word)
                fw.write((' '.join(key_word) + '\t' + ' '.join(text) + '\n').encode('utf-8'))
                count += 1

            if count % 100 == 0:
                print(count)
    print("Total:%d; Now:%d" % (i, count))
    print("Split into three")
    split_into_three(src_path=dst_path, dst_path='../data/topic_to_review/')


def get_sentiment_transfer_data(src_path, dst_paths):
    with open(src_path) as f, open(dst_paths[0], 'w') as fw1, open(dst_paths[1], 'w') as fw2:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for i, row in enumerate(f_csv):
            review = row[1].strip().decode('utf-8')
            senti = row[2]
            words = word_tokenize(review)
            words = [word.lower() for word in words]
            review = ' '.join(words).encode('utf-8')

            if senti == "Positive":
                fw1.write(review + '\n')
            if senti == "Negative":
                fw2.write(review + '\n')

    print("Split data!")
    for i in range(2):
        split_into_three(src_path=dst_paths[i], dst_path='../data/', label='.%d' % i)


def split_into_three(src_path, dst_path, label=''):
    lines = open(src_path).readlines()
    np.random.shuffle(lines)

    with open(os.path.join(dst_path, 'train' + label), 'w') as f:
        for line in lines[:-1000]:
            f.write(line)
    with open(os.path.join(dst_path, 'dev' + label), 'w') as f:
        for line in lines[-1000:-500]:
            f.write(line)
    with open(os.path.join(dst_path, 'test' + label), 'w') as f:
        for line in lines[-500:]:
            f.write(line)


def get_test_key_words(src_path, dst_path):
    words = []
    with open(src_path) as f:
        lines = f.readlines()
        for line in lines:
            words.append(line.strip())

    com_words = words[1:23]
    spt_words = words[25:]

    print(com_words[-1], spt_words[0])

    with open(dst_path, 'w') as f:
        for i in range(100):
            for j in range(3, 6):
                words = random.sample(com_words, j)
                f.write(' '.join(words) + '\n')
        for i in range(100):
            for j in range(3, 6):
                words = random.sample(spt_words, j)
                f.write(' '.join(words) + '\n')


if __name__ == "__main__":
    review_path = '../data/googleplaystore_user_reviews.csv'
    get_topic_to_review_data(src_path=review_path,
                             dst_path='../data/topic_to_review/big.txt')
    
    get_sentiment_transfer_data(src_path=review_path,
                                dst_paths=['../data/sentiment_transfer/neg',
                                           '../data/sentiment_transfer/pos'])



