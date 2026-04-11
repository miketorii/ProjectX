import os

from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

#from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
    
######################################
#
#    
print("----------start----------------")

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

agent = create_csv_agent(
    model,
    "data/heart_disease_uci.csv",
    verbose=True,
    allow_dangerous_code=True
)

#response = agent.invoke("How many rows of data are in the file?")
#print(response)

response = agent.invoke("What are the columns within the dataset?")
print(response)

#response = agent.invoke("データセットの中のカラムは何ですか？")
#print(response)

print("-------------------------------")
print("----------end------------------")





