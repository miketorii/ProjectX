from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModel, AutoModelForSequenceClassification
import torch

classifier = pipeline("sentiment-analysis")
raw_inputs =  [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
]

# Tokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

#Model
model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

#Model for SequenceClassification
model2 = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs2 = model2(**inputs)
print(outputs2.logits.shape)
print(outputs2.logits)

#Softmax using torch
predictions = torch.nn.functional.softmax(outputs2.logits)
print(predictions)
result = model2.config.id2label
print(result)

ret = classifier(
    raw_inputs
)

print(ret)
