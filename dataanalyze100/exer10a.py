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
