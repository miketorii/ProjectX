import asyncio
import os
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

from time import sleep

load_dotenv()


FONDRY_MODE=os.getenv('FONDRY_NAME')
FONDRY_PROJECT_ENDPOINT=os.getenv("FONDRY_PROJECT_ENDPOINT")
AGENT_NAME=os.getenv("AGENT_NAME2")

async def main() -> None:
    print('---IN main--')

    project = AIProjectClient(
        endpoint=os.environ["FONDRY_PROJECT_ENDPOINT"],        
        credential=DefaultAzureCredential(),                
    )
    openai = project.get_openai_client()

    response = openai.responses.create(
        extra_body={"agent_reference": {"name": AGENT_NAME, "type": "agent_reference"} },
        input="What is the largest city in France?",
        background=True,
    )

    while response.status in ("queued", "in_progress"):
        sleep(2)
        response = openai.responses.retrieve(response.id)
    
    print(response.output_text)
    

if __name__ == "__main__":
    print('---------start------------')
    asyncio.run(main())
    

