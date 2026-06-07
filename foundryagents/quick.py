import asyncio
import os
from dotenv import load_dotenv

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential

load_dotenv()

FONDRY_MODE=os.getenv('FONDRY_NAME')
FONDRY_PROJECT_ENDPOINT=os.getenv("FONDRY_PROJECT_ENDPOINT")

async def main() -> None:
    print('---IN main--')

    agent = Agent(
        client=FoundryChatClient(
            project_endpoint=os.environ["FONDRY_PROJECT_ENDPOINT"],
            model=os.environ["FONDRY_MODEL"],
            credential=AzureCliCredential(),
        ),
        instructions="you are a helpful assistant.",
    )

    result = await agent.run("What is the capital of France?")
    print(f"Agent: {result}")

if __name__ == "__main__":
    print('---------start------------')
    asyncio.run(main())
    

