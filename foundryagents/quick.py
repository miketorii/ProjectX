import asyncio
import os
from dotenv import load_dotenv

from random import randint
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential

from pydantic import Field


load_dotenv()

FONDRY_MODE=os.getenv('FONDRY_NAME')
FONDRY_PROJECT_ENDPOINT=os.getenv("FONDRY_PROJECT_ENDPOINT")

@tool(approval_mode="never_require")
def get_weather(
        locaiton: Annotated[str, Field(description="The location to get the weather for.")]
) -> str:
    conditions = ["sunny", 'cloudy', "rainy", "stormy"]
    return f"The weather in {locaiton} is {conditions[randint(0,3)]} with a high of {randint(10,30)}C."

async def main() -> None:
    print('---IN main--')

    agent = Agent(
        client=FoundryChatClient(
            project_endpoint=os.environ["FONDRY_PROJECT_ENDPOINT"],
            model=os.environ["FONDRY_MODEL"],
            credential=AzureCliCredential(),
        ),
        instructions="you are a helpful assistant.",
        tools=get_weather,
    )

#    question = "What is the capital of France?"
    question = "What is the weather like in Seattle?"        
    result = await agent.run(question)
    print(f"Agent: {result}")

if __name__ == "__main__":
    print('---------start------------')
    asyncio.run(main())
    

