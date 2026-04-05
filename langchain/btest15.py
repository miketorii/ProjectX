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



print("-------end----------")

