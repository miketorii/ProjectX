import asyncio
import os
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

load_dotenv()

FONDRY_MODE=os.getenv('FONDRY_NAME')
FONDRY_PROJECT_ENDPOINT=os.getenv("FONDRY_PROJECT_ENDPOINT")
AGENT_NAME=os.getenv("AGENT_NAME")

async def main() -> None:
    print('---IN main--')

    project = AIProjectClient(
        endpoint=os.environ["FONDRY_PROJECT_ENDPOINT"],        
#        credential=AzureCliCredential(),
        credential=DefaultAzureCredential(),                
    )

    agent = project.agents.create_version(
        agent_name=AGENT_NAME,
        definition=PromptAgentDefinition(
            model=os.environ["FONDRY_MODEL"],
        instructions="you are a helpful assistant that answers general questions.",            
        )
    )

    print(f"Agent created id: {agent.id}, name: {agent.name}, version: {agent.version}")

if __name__ == "__main__":
    print('---------start------------')
    asyncio.run(main())
    

