import os
from openai import AzureOpenAI

from langchain_openai import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal, Union

import json

##################################
#
#
def schedule_meeting(date, time, attendees):
    return { "event_id": "1234",
             "status": "Meegint scheduled successfully!",
             "date": date,
             "time": time,
             "attendees": attendees,             
            }

OPENAI_FUNCTIONS = {
    "schedule_meeting": schedule_meeting
}

######################################
#
#
functions = [
    {
        "type": "function",
        "function": {
            "type":"object",
            "name":"schedule_meeting",
            "description":"Set a meeting at a specified date and time for designated attendees",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "format": "date"},
                    "time": {"type": "string", "format": "time"},
                    "attendees": {"type":"array", "items": {"type":"string"} },                    
                },
                "required": ["date", "time", "attendees"],
            },            
        }
    }
]

######################################
#
#    
print("----------start----------------")

#model = AzureChatOpenAI(
#    azure_deployment="MyModel",  # or your deployment
#    api_version="2024-02-01",  # or your api version
#    temperature=0,
#    max_tokens=None,
#    timeout=None,
#    max_retries=2,
    # other params...
#)

client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-08-01-preview"
)

messages = [
    {
        "role": "user",
        "content": "Schedule a meeting on 2026-5-1 at 14:00 with Alice and Bob",
    }
]

response = client.chat.completions.create(
    model="my-gpt-4o-1",
    messages=messages,
    tools=functions,
    tool_choice="auto"
)

print(response.choices[0].message.content)

response = response.choices[0].message

print("========================================")

if response.tool_calls:
    first_tool_call = response.tool_calls[0]

    function_name = first_tool_call.function.name
    function_args = json.loads(first_tool_call.function.arguments)
    print("Function name: ", function_name)
    print("Arguments: ", function_args)

    function = OPENAI_FUNCTIONS.get(function_name)

    function_response = function(**function_args)

    messages.append(
        {
            "role": "function",
            "name": "schedule_meeting",
            "content": json.dumps(function_response),
        }
    )

    second_response = client.chat.completions.create(
        model="my-gpt-4o-1",
        messages=messages)

    print(second_response.choices[0].message.content)
    
print("-------------------------------")
print("----------end------------------")





