import os
import urllib.request
import re
from token import SimpleTokenizerV1
from tokenv2 import SimpleTokenizerV2

if not os.path.exists("the-verdict.txt"):
#    url = ("https://github.com/rasbt/"
#           "LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/"
#           "the-verdict.txt" )
    file_path = "the-verdict.txt"

    url = "https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/the-verdict.txt"

    print(url)
    print(file_path)
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt","r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total: ", len(raw_text))
print(raw_text[:99])

print("---------Token-----------")
preprocessed = re.split(r'([,.:;?_!()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 30:
        break

print("-------------------tokenizer v1--------------------")    
tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know,"
           Mr.s. Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)
print(ids)

result = tokenizer.decode(ids)
print(result)

#########################################################3
#

print("-------------------unknown token--------------------")

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}

vocab_size = len(vocab.items())
print(vocab_size)

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

print("-------------------tokenizer v2--------------------")

tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1,text2))
print(text)

ids = tokenizer.encode(text)
print(ids)

result = tokenizer.decode(ids)
print(result)
