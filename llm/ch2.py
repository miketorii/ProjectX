import os
import urllib.request
import re

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
result = re.split(r'([,.:;?_!()\']|--|\s)', raw_text)
result = [item for item in result if item.strip()]
print(len(result))
print(result[:30])
