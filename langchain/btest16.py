from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from typing import Optional

import os
import json
from os import getenv

print("-------start----------")

messages = [
    {
        "role": "user",
        "content":"Schedule a meeting on 2023-11-01 at 14:00 with Alice and Bob."
    }
]


functions = [
    {
        "type": "function",
        "function": {
            "type": "object",
            "name": "schedule_meeting",
            "description": "Set a meeting at a specified date and time for designated attendees",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "format": "date"},
                    "time": {"type": "string", "format": "time"},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["date", "time", "attendees"],
            },
        },
    }
]

#model = AzureChatOpenAI(
#    azure_deployment="MyModel",  # or your deployment
#    api_version="2024-02-01",  # or your api version
#    temperature=0,
#    max_tokens=None,
#    timeout=None,
#    max_retries=2,
    # other params...
#)

model = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-08-01-preview"
)

response = model.chat.completions.create(
    model = "my-gpt-4o-1",
    messages=messages,
    tools=functions,
    tool_choice="auto"
)

response = response.choices[0].message

print(response)

print("-------end----------")

