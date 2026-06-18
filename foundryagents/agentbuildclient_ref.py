import asyncio
import os
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

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
        store=False,
    )
    print(response.output_text)

    response2 = openai.responses.create(
        extra_body={"agent_reference": {"name": AGENT_NAME, "type": "agent_reference"} },
        input=[
            {"role": "user", "content": "What is the largest city in France?"},
            {"role": "assistant", "content": response.output_text},
            {"role": "user", "content": "What is the population of that city?"}
        ],
        store=False,
    )
    print(response2.output_text)
    

if __name__ == "__main__":
    print('---------start------------')
    asyncio.run(main())
    

