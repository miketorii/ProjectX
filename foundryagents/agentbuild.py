import asyncio
import os
from dotenv import load_dotenv

from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, WebSearchTool

load_dotenv()

FONDRY_MODE=os.getenv('FONDRY_NAME')
FONDRY_PROJECT_ENDPOINT=os.getenv("FONDRY_PROJECT_ENDPOINT")

async def main() -> None:
    print('---IN main--')

    project = AIProjectClient(
        endpoint=FONDRY_PROJECT_ENDPOINT,
        credential=DefaultAzureCredential()
#        credential=AzureCliCredential()
    )
    
    agent = project.agents.create_version(
        agent_name="mike-agent-20260613",
        definition=PromptAgentDefinition(
            model="gpt-4o",
            instructions="you are a helpful assistant.",
        )
    )

    print(f"Agent: {agent.name}, Version: {agent.version}")

if __name__ == "__main__":
    print('---------start------------')
    asyncio.run(main())
    

