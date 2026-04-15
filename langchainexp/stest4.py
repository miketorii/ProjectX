from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

#import os

from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

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

db = SQLDatabase.from_uri("sqlite:///./data/demo.db")
toolkit = SQLDatabaseToolkit(db=db, llm=model)

agent_executor = create_sql_agent(
    llm=model,
    toolkit=toolkit,
    verbose=True,
    # AgentType.OPENAI_TOOLS の代わりに文字列で指定
    agent_type="openai-tools" 
)

result = agent_executor.invoke("Identify all of the tables")
print(result)

print("-------------------------------")
print("----------end------------------")





