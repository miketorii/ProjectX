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

#################### 96 ##########################

