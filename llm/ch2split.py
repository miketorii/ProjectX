import re

#text = "Hello, world. This, is a test."
text = "Hello, world. Is this-- a test?"
#result = re.split(r'(\s)', text)
#result = re.split(r'[,.]|\s', text)
result = re.split(r'([,.:;?_!()\']|--|\s)', text)
result = [item for item in result if item.strip()]
print(result)

