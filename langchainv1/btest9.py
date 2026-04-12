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


class Query(BaseModel):
    id: int
    question: str
    dependencies: List[int] = Field(
        default_factory=list,
        description="""A list of sub-queries that must be completed before this task can be completed. Use a sub query when anythin is unknown and we might need to ask many queries to get an answer. Dependencies must only be other queries.
        """)    

class QueryPlan(BaseModel):
    query_graph: List[Query]
    
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

parser = PydanticOutputParser(pydantic_object=QueryPlan)

template = """Generate a query plan. This will be used for task execution.

Answer the following query: {query}

Return the following query graph format:
{format_instructions}
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

chain = chat_prompt | client | parser

result = chain.invoke(
    {"query":'''I want to get the results from my database. Then I want to find out what the average age of my top 10 customers is. Once I have the average age, I want to send an email to John. Also I just generally want to send a welcome introduction email to Sara regardless of the other tasks.
    ''',
     "format_instructions": parser.get_format_instructions()
    }
)

print(result.query_graph)


print("-------------------------------")
print("----------end------------------")





