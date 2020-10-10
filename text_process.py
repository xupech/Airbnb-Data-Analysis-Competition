#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:10:08 2019

@author: xupech
"""

import csv
import os
import math
import re
import string
import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import nltk
from nltk.probability import FreqDist
from nltk.text import Text as text1
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import figure

listing = pd.read_csv(r'/Users/xupech/Desktop/brandeis graduate school/2019 Data Competition/Data Sets/listings_details.csv', low_memory = False)

# get rid of all emojis in comments; collect results in a list with index.


price=listing['price']

""" name only, templist0 returns tokenized names, name_length returns length of each name."""
""" name only, templist1 returns tokenized names, desc_length returns length of each description."""



    
templist0={}
name_length = {}

templist1={}
desc_length=[]
for index, row in listing.iterrows():
    temp = str(row['name'])
    temp2 = str(row['description'])
    if len(str(temp))>0:
        temp.encode('ascii', 'ignore').decode('ascii')
        temp2.encode('ascii', 'ignore').decode('ascii')
        
        temp1 = nltk.word_tokenize(temp)
        temp3 = nltk.word_tokenize(temp2)
        length = len(temp1)
        length0 = len(temp3)
    else:
        temp1 = None
        temp3 = None
    templist0.update({index: temp1})
    name_length.update({index: length})
    
    templist1.update({index: temp3})
    desc_length.append(length0)

dummy_bigram=pd.DataFrame(desc_length, columns=['desc_length'])
    
""" combining name, summary, space, description together, templist returns tokenized thing """
""" vocab is the vocaburary for above four columns, including punctuations/content words"""
templist={}
vocab=[]
for index, row in listing.iterrows():
    temp = str(row['name'])+' '+str((row['summary']))+' '+str((row['space']))+' '+str((row['description']))
    if len(str(temp))>0:
        temp.encode('ascii', 'ignore').decode('ascii')
        temp1 = nltk.word_tokenize(temp)
    else:
        temp1 = None
    templist.update({index: temp1})
    vocab+=temp1

""" return top 50 common non-content words used in the four columns combined """ 
STOPLIST = set(nltk.corpus.stopwords.words())
def is_content_word(word):
    return word.lower() not in STOPLIST and word[0].isalpha()


dist = nltk.FreqDist([w.lower() for w in vocab if is_content_word(w)])
freq2=dist.most_common(50)
# Oops, the words here are not informative. I will try bigrams instead.

""" bigrams, b_dict returns a dictionary of bigrams each row; b_vocab gives the whole bigrams vocaburary """
b_dict={}
bivocab=[]
for index, row in templist.items():
    filtered_temp =[b for b in list(nltk.bigrams(row)) if is_content_word(b[0]) 
                    and is_content_word(b[1])]
    b_dict.update({index: filtered_temp})
    bivocab+=filtered_temp

dist1 = nltk.FreqDist([b for b in bivocab])
freq0 = dist1.most_common(50)


biig, biigfreq=zip(*freq0)

fig, ax = plt.subplots()
index = np.arange(len(biig))
bar_width = 0.25
opacity = 0.8

chart0 = ax.barh(index, biigfreq, bar_width, align = 'center', alpha=0.5)
for i, v in enumerate(biigfreq):
    ax.text(v+3, i, str(v), color='blue')

plt.yticks(index, biig)
plt.ylabel('Bigrams')
plt.title('Most Popular Bigrams In Name, Summary, Space And Description Columns Conbined')

plt.show()

fig.savefig('Most Popular Bigrams In Name, Summary, Space And Description Columns Conbined')

dist_desc=[]
listing_desc=[]
for key, value in b_dict.items():
    for bigram in value:
        n=0
        if bigram in biig:
            n+=1
    
        else:
            continue
    if n == 0:
        listing_desc.append(0)
        dist_desc.append(0)
    else:
        listing_desc.append(1)
        dist_desc.append(n)
        

dummy_bigram['listing_desc']=listing_desc 


""" neighborhood and transit combined, replicate the process to have each line tokenized, 
    create a vocaburary, get the non-content words distribution and bigrams distribution. """

templist2 = {}
tranvo = []
tb_dict={}
tbvocab=[]
for index, row in listing.iterrows():
    temp = str(row['neighborhood_overview'])+' '+str((row['transit']))
    if len(str(temp))>0 and temp != 'nan nan':
        temp.encode('ascii', 'ignore').decode('ascii')
        temp1 = nltk.word_tokenize(temp)
        templist2.update({index: temp1})
        tranvo+=temp1
    else:
        templist2.update({index: None})
    

""" return top 50 common non-content words in neighborhood and transit combined."""
dist2 = nltk.FreqDist([w.lower() for w in tranvo if is_content_word(w)])
freqtran=dist2.most_common(50)


for index, value in templist2.items():
    if value != None:
        filtered_temp =[b for b in list(nltk.bigrams(value)) if (is_content_word(b[0]) 
                        and is_content_word(b[1]))]
        tb_dict.update({index: filtered_temp})
        tbvocab+=filtered_temp
    else:
        tb_dict.update({index: None})

distbit = nltk.FreqDist([b for b in tbvocab])
freqbit = distbit.most_common(50)


biit, biitfreq=zip(*freqbit)

fig, ax = plt.subplots()
index = np.arange(len(biit))
bar_width = 0.15
opacity = 0.8

chart0 = ax.barh(index, biitfreq, bar_width, align = 'center', alpha=0.5)
for i, v in enumerate(biitfreq):
    ax.text(v+3, i, str(v), color='blue')

plt.yticks(index, biit)
plt.ylabel('Bigrams')
plt.title('Most Popular Bigrams In Neighborhood And Transit Columns Conbined')

plt.show()
fig.savefig('Most Popular Bigrams In Neighborhood And Transit Columns Conbined')

location=[]
dist_location=[]
for key, value in tb_dict.items():
    if value != None:
        for bigram in value:
            n=0
            if bigram in biit:
                n+=1
    
            else:
                continue
        if n == 0:
            location.append(0)
            dist_location.append(0)
        else:
            location.append(1)
            dist_location.append(n)
    else:
        location.append(0)
        dist_location.append(0)

dummy_bigram['location']=location 


review = pd.read_csv(r'/Users/xupech/Desktop/brandeis graduate school/2019 Data Competition/Data Sets/reviews_details.csv', low_memory = False)
review_token={}
review_length={}
review_ave_length=[]
review_number ={}
reviewv = []
review_bigram ={}
rbivo =[]

""" generate review_token as a dict with listing_id as keys and all tokens in the listing as values. """
""" generate review_length to record length of each review in a list, list is stored as values in a dict. """
""" generate review_number as a dict to record number of reviews per listing."""
""" generate a reviewv for all vocaburary in reviews. """


for index, row in review.iterrows():
    tempin = int(row['listing_id'])
    temp = str(row['comments'])
    if len(str(temp))>0:
        temp.encode('ascii', 'ignore').decode('ascii')
        temp1 = nltk.word_tokenize(temp)
        templen = [len(temp1)]
    else:
        temp1 = None
        templen = 0
        
    if tempin not in review_token.keys():
        review_token.update({tempin: temp1})

    else:
        review_token[tempin]+=temp1
     
    if tempin not in review_length.keys():
        review_length.update({tempin: templen})
    else:
        review_length[tempin]+=templen
    
    reviewv+=temp1


for index, value in review_length.items():
    temp = len(value)
    review_number.update({index: temp})
    review_ave_length.append(int(np.mean(value)))    
    

# most frequent words.  
distrev = nltk.FreqDist([w.lower() for w in reviewv if is_content_word(w)])
freqrev=distrev.most_common(50)

for index, row in review_token.items():
    filtered_temp =[b for b in list(nltk.bigrams(row)) if is_content_word(b[0]) 
                    and is_content_word(b[1])]
    
    review_bigram.update({index: filtered_temp})
    rbivo+=filtered_temp

distbrt = nltk.FreqDist([b for b in rbivo])
freqbrt = distbrt.most_common(50)        


brt, brtfreq=zip(*freqbrt)

fig, ax = plt.subplots()
index = np.arange(len(brt))
bar_width = 0.15
opacity = 0.8

chart0 = ax.barh(index, brtfreq, bar_width, align = 'center', alpha=0.5)
for i, v in enumerate(brtfreq):
    ax.text(v+3, i, str(v), color='blue')

plt.yticks(index, brt)
plt.ylabel('Bigrams')
plt.title('Most Popular Bigrams In Comments')

plt.show()
fig.savefig('Most Popular Bigrams In Comments')


review_dummy=[]

dist_review=[]
for key, value in review_bigram.items():
    for bigram in value:
        n=0
        if bigram in brt:
            n+=1
    
        else:
            continue
    if n == 0:
        review_dummy.append(0)
        dist_review.append(0)
    else:
        review_dummy.append(1)
        dist_review.append(n)
        
reviewdf = pd.DataFrame(list(review_number.items()), columns=['listing index','number of reviews'])
reviewdf['average length of reviews per listing']=review_ave_length
reviewdf['review_dummy']=review_dummy





reviewdf.to_csv('review_text_prcessed.csv', index=False)

dummy_bigram.to_csv('listing_dummy and location_dummy.csv', index=False)


del dummy_bigram['std index']













