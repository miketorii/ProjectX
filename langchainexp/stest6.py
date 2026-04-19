import os
import re
#from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate

from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

######################################
#
#    
print("----------start----------------")

text = """
Action: search_on_google
Action_Input: Tom Hanks current wife

action: search_on_wikipedia
action_input: How old is Rita Wilson in 2023

action: search_on_google
action input: some other query
"""

action_pattern = re.compile(r"(?i)action\s*:\s*([^\n]+)", re.MULTILINE)
action_input_pattern = re.compile(r"(?i)action\s*_*input\s*:\s*([^\n]+)", re.MULTILINE)

actions = action_pattern.findall(text)
action_inputs = action_input_pattern.findall(text)

print(actions)
print(action_inputs)

last_action = actions[-1] if actions else None
last_action_input = action_inputs[-1] if action_inputs else None

print("Last Action: ", last_action)
print("Last Action Input: ", last_action_input)


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


print("-------------------------------")
print("-------------------------------")
print("----------end------------------")





