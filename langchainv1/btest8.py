import os
from openai import AzureOpenAI

from langchain_openai import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

from langchain.chains.openai_tools import create_extraction_chain_pydantic

##################################
#
#


class Person(BaseModel):
    name: str = Field(description="人の名前")
    age: int = Field(description="人の年齢")    

######################################
#
#    
print("----------start----------------")

client = AzureChatOpenAI(
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

chain = create_extraction_chain_pydantic(Person, client)
result = chain.invoke(
    {"input":'''
    Bob is 25 years old. He lives in New York. He likes to play basketball. 
    Sara is 30 years old. She lives in San Francisco. She likes to play tennis.
    Tom is 18 years old. He lives in Tokyo. He likes to play baseball.'''}
)

print(result)

print("-------------------------------")
print("----------end------------------")





