import os
from openai import AzureOpenAI

from langchain_openai import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate

from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List

from langchain.chains.openai_tools import create_extraction_chain_pydantic

import pickle

##################################
#
#
print("----------start----------------")

examples = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
    },
    {
        "question": "What is the capital of Spain?",
        "answer": "Madrid",
    },
    {
        "question": "What is the capital of Germany?",
        "answer": "Berlin",
    },
    {
        "question": "What is the capital of England?",
        "answer": "London",
    },    
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human","{question}"),
        ("ai","{answer}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

print(few_shot_prompt.format())

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a responsible for answering questions about countries. Only return the country name"
        ),
        few_shot_prompt,
        ("human","{question}"),
    ]
)

model = AzureChatOpenAI(
    azure_deployment="MyModel",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

#client = AzureOpenAI(
#    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
#    api_key=os.getenv("AZURE_OPENAI_KEY"),
#    api_version="2024-08-01-preview"
#)

chain = final_prompt | model | StrOutputParser()

result = chain.invoke(
    {
        "question": "What is the capital of United States?"
    }
)

print(result)

with open('few_shot_prompt.pickle', 'wb') as f:
    pickle.dump(few_shot_prompt, f)

with open('few_shot_prompt.pickle', 'rb') as f:
    few_shot_prompt = pickle.load(f)
    print(few_shot_prompt)
    
print("-------------------------------")
print("----------end------------------")





