##################### 94 ########################
## !pip install mecab-python3 unidic-lite

import MeCab

tagger = MeCab.Tagger()
text = "すもももももももものうち"
words = tagger.parse(text)
print(words)

words = tagger.parse(text).splitlines()
words_arr = []
for i in words:
    if i == 'EOS': continue
    word_tmp = i.split()[0]
    words_arr.append(word_tmp)

print(words_arr)

#################### 95 ##########################

text2 = "すもももももももものうち"
words2 = tagger.parse(text2).splitlines()
words_arr2 = []
parts = ["名詞","動詞"]
for i in words2:
    if i == 'EOS' or i == '' : continue
    word_tmp = i.split()[0]
    part = i.split()[4].split("-")[0]
    if not (part in parts) : continue
    words_arr2.append(word_tmp)

print(words_arr2)

#################### prep ##########################

import os
from google.colab import drive
import pandas as pd

drive.mount("/content/drive")

survey = pd.read_csv("/content/drive/MyDrive/chap10/survey.csv")
print( survey.head() )

#################### 96 ##########################

all_words = []
parts = ["名詞"]

#text2 = "hey guys"
for n in range(len(survey)):
  text = survey["comment"].iloc[n]
  #print(text)
  words = tagger.parse(str(text)).splitlines()
  words_arr3 = []
  for i in words:
    if i == "EOS" or i == "" : continue
    word_tmp = i.split()[0]
    if len(i.split()) >= 4:
      part = i.split()[4].split("-")[0]
      if not (part in parts) : continue
      words_arr3.append(word_tmp)
  all_words.extend(words_arr3)

print(all_words)

all_words_df = pd.DataFrame( { "words": all_words, "count": len(all_words)*[1] })
all_words_df = all_words_df.groupby("words").sum()
print( all_words_df.sort_values("count", ascending=False).head(20) )

#################### 97 ##########################

stop_words = ["時"]
all_words4 = []
parts = ["名詞"]

for n in range(len(survey)):
  text = survey["comment"].iloc[n]
  words = tagger.parse(str(text)).splitlines()
  words_arr4 = []
  for i in words:
    if i == "EOS" or i == "" : continue
    word_tmp4 = i.split()[0]
    if len(i.split()) >= 4:
      part = i.split()[4].split("-")[0]
      if not (part in parts) : continue
      if word_tmp4 in stop_words: continue
      words_arr4.append(word_tmp4)
  all_words4.extend(words_arr4)

print(all_words4)

all_words_df4 = pd.DataFrame({"words": all_words4, "count": len(all_words4)*[1]})
all_words_df4 = all_words_df4.groupby("words").sum()
print( all_words_df4.sort_values("count", ascending=False).head(20) )

#################### 98 ##########################

stop_words = ["時"]
parts = ["名詞"]
all_words5 = []
satisfaction = []

for n in range(len(survey)):
  text = survey["comment"].iloc[n]
  words = tagger.parse(str(text)).splitlines()
  words_arr5 = []
  for i in words:
    if i == "EOS" or i == "" : continue
    word_tmp5 = i.split()[0]
    if len(i.split()) >= 4:
      part = i.split()[4].split("-")[0]
      if not (part in parts) : continue
      if word_tmp5 in stop_words: continue
      words_arr5.append(word_tmp5)
      satisfaction.append(survey["satisfaction"].iloc[n])
  all_words5.extend(words_arr5)

print(all_words5)

all_words_df5 = pd.DataFrame({"words": all_words5, "satisfaction": satisfaction, "count": len(all_words5)*[1]})
print( all_words_df5.head() )

words_satisfaction = all_words_df5.groupby("words").mean()["satisfaction"]
words_count = all_words_df5.groupby("words").sum()["count"]
words_df = pd.concat([words_satisfaction, words_count],axis=1)
print( words_df.head() )

#################### 99 #######################

parts = ["名詞"]
all_words_df = pd.DataFrame()
satisfaction = []

for n in range(len(survey)):
  text = survey["comment"].iloc[n]
  words = tagger.parse(str(text)).splitlines()
  words_df = pd.DataFrame()
  for i in words:
    if i == "EOS" or i == "" : continue
    word_tmp = i.split()[0]
    if len(i.split()) >= 4:
      part = i.split()[4].split("-")[0]
      if not (part in parts) : continue
      words_df[word_tmp] = [1]
  all_words_df = pd.concat([all_words_df, words_df], ignore_index=True)

#print(all_words_df.head())
all_words_df = all_words_df.fillna(0)
print(all_words_df.head())

################## 100 #################

print(survey["comment"].iloc[2])
target_text = all_words_df.iloc[2]
print(target_text)

import numpy as np

cos_sim = []
for i in range(len(all_words_df)):
  cos_text = all_words_df.iloc[i]
  cos = np.dot(target_text, cos_text) / ( np.linalg.norm(target_text) * np.linalg.norm(cos_text) )
  cos_sim.append(cos)

all_words_df["cos_sim"] = cos_sim
print( all_words_df.sort_values("cos_sim",ascending=False).head() )

print(survey["comment"].iloc[2])
print(survey["comment"].iloc[24])
print(survey["comment"].iloc[15])
print(survey["comment"].iloc[33])


