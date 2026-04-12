import os
from openai import AzureOpenAI

from langchain_openai import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate, FewShotPromptTemplate, PromptTemplate

from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List

from langchain.chains.openai_tools import create_extraction_chain_pydantic

from langchain.prompts.example_selector import LengthBasedExampleSelector

import pickle
import tiktoken

##################################
#
#

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

##################################
#
#
print("----------start----------------")

examples = [
    {"input": "Gollum", "output": "<Story involving Gollum>" },
    {"input": "Gandalf", "output": "<Story involving Gandalf>" },
    {"input": "Bilbo", "output": "<Story involving Bilbo>" },    
]

story_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Character: {input}\nStory: {output}",
)

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=story_prompt,
    max_length=1000,
    get_text_length=num_tokens_from_string,
)

dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=story_prompt,
    prefix="Generate a story for {character} using the current Character/Story pairs from all of the characters as context.",
    suffix="Character: {character}\nStory:",
    input_variables=["character"],
)

formatted_prompt = dynamic_prompt.format(character="Frodo")

print(formatted_prompt)

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

result = model.invoke(
    [SystemMessage(content=formatted_prompt)]
)

print(result.content)

print("-------------------------------")
print("----------end------------------")





