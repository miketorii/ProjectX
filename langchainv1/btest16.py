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

def schedule_meeting(date, time, attendees):
    # Connect to calendar service:
    return { "event_id": "1234", "status": "Meeting scheduled successfully!",
            "date": date, "time": time, "attendees": attendees }

OPENAI_FUNCTIONS = {
    "schedule_meeting": schedule_meeting
}

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

if response.tool_calls:
    for tool_call in response.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        print(function_name)
        print(function_args)

        function = OPENAI_FUNCTIONS.get(function_name)

        if not function:
            raise Exception(f"Function {function_name} not found.")

        function_response = function(**function_args)

        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": json.dumps(function_response),
            }
        )

    second_response = model.chat.completions.create(
        model = "my-gpt-4o-1",
        messages=messages
    )

    print(second_response.choices[0].message.content)

print("-------end----------")

