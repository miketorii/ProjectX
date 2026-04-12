import os
from openai import AzureOpenAI

from langchain_openai import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal, Union, Optional

import json

from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.output_parsers.openai_tools import PydanticToolsParser

##################################
#
#
class Article(BaseModel):
    points: str = Field(..., description="記事の要点")
    contrarian_points: Optional[str] = Field(None, description="記事で述べられている反対意見")
    author: Optional[str] = Field(None, description="記事の著者")
    
_EXTRACTION_TEMPLATE = """
以下の文章で言及されている関連のエンティティをその属性とともに抽出し保存してください。
プロパティが存在せず、関数のパラメーターに必要ない場合は、出力に含めないでください。
"""

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

#model = AzureOpenAI(
#    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
#    api_key=os.getenv("AZURE_OPENAI_KEY"),
#    api_version="2024-08-01-preview"
#)

prompt = ChatPromptTemplate.from_messages(
    {("system", _EXTRACTION_TEMPLATE),
     ("user","{input}")
     }
)

pydantic_schemas = [Article]

tools = [convert_to_openai_tool(p) for p in pydantic_schemas]

model = model.bind_tools(tools=tools)

chain = prompt | model | PydanticToolsParser(tools=pydantic_schemas)

result = chain.invoke(
    {

        "input": """In the recent article titled 'AI adoption in industry', key points addressed include the growing interest in AI in various sectors, the increase in AI research, and the need for responsible AI. However, the author, Dr. Jane Smith, acknowledges a contrarian view — that without stringent regulations, AI may pose high risks."""
    }
)

print(result)

print("-------------------------------")
print("----------end------------------")

# "input" : """「産業におけるAIの採用」と題された最近の記事では、要点として、次のようなことに対する高まる関心が取り上げられています... しかし、著者のジェーン・スミス博士は... """
        





