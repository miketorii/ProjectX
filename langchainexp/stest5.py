from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

#import os

#from langchain.agents import create_react_agent
#from langchain.agents.react.agent import create_react_agent
#from langchain.agents import create_react_agent
#from langchain.agents.agent import AgentExecutor

from langgraph.prebuilt import create_react_agent
#from langchain import hub
from langsmith import Client
#from langchain.hub import pull
from langchain.tools import Tool

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

user_sql = agent_executor.invoke("Add 5 new users to the database. Their names are: Jeff, Mike, Tom, James and Devon. Run the following SQL query against the database and add the users.")
print(user_sql)

print("-------------------------------")

SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
If the question does not seem related to the database, just return "I don't know" as the answer.
"""
agent_executor2 = create_sql_agent(
    llm=model,
    toolkit=toolkit,
    verbose=True,
    # AgentType.OPENAI_TOOLS の代わりに文字列で指定
    agent_type="openai-tools",
    prefix=SQL_PREFIX.format(dialect="SQLite", top_k=100)
)

ret = agent_executor2.invoke(user_sql)
print(ret)

ret = agent_executor2.invoke("Do we have a Peter in the database")
print(ret)

ret = agent_executor2.invoke("Do we have a Bob in the database")
print(ret)

print("----------end------------------")





