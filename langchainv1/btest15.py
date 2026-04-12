from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from typing import Optional

import json
from os import getenv

class Article(BaseModel):
    points: str = Field(..., description="Key points from the article")
    constrain_points: Optional[str] = Field( None, description="Any constrain points acknowledge in the article")
    author: Optional[str] = Field( None, description="Author of the article")

    
print("-------start----------")

_EXTRACTION_TEMPLATE = """
Extract and save the relevant entities mentiond
in the following passage together with their properies.

if a property is not present and is not required in the function parameters,
do not include it in the outpu.
"""

prompt = ChatPromptTemplate.from_messages(
    {
        ("system",_EXTRACTION_TEMPLATE),
        ("user","{input}")
    }
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

pydantic_schemas = [Article]

tools = [convert_to_openai_tool(p) for p in pydantic_schemas]

#for p in pydantic_schemas:
#    print(p)
[print(p) for p in pydantic_schemas]

model = model.bind_tools(tools=tools)

chain = prompt | model | PydanticToolsParser(tools=pydantic_schemas)

result = chain.invoke(
    {
        "input":"""In the recent article titled 'AI adaption in industry', key points addressed include the glowing interest in AI in various sectors, the increase in AI research, and the need for responsible AI. However, the author, Dr. Jane Smith, acknowledges a contrarian view --- that without stringent regulations, AI may pose high risks."""
    }
)

print(result)

print("-------end----------")

