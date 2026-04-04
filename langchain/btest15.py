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



print("-------end----------")

