import asyncio
import os
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

from azure.ai.projects.models import MemoryStoreDefaultDefinition, MemoryStoreDefaultOptions


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

    options = MemoryStoreDefaultOptions(
        chat_summary_enabled=True,
        user_profile_enabled=True
    )
    definition = MemoryStoreDefaultDefinition(
        chat_model="gpt-4o",
        embedding_model="text-embedding-3-small-2",
        options=options
    )
    memory_store = project.beta.memory_stores.create(
        name="my_memory_store",
        definition=definition,
        description="Memory store for my agent"
    )
    print(f"Memory store: {memory_store.name}")

    response = openai.responses.create(
        extra_body={"agent_reference": {"name": AGENT_NAME, "type": "agent_reference"} },
        input="What is the largest city in France?",
#        store=False,
    )
    print(response.output_text)


if __name__ == "__main__":
    print('---------start------------')
    asyncio.run(main())
    

