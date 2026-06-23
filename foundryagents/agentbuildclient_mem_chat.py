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

CHAT_MODEL_NAME=os.getenv("CHAT_MODEL_NAME")
EMBEDDING_MODEL_NAME=os.getenv("EMBEDDING_MODEL_NAME")

MEMORY_STORE_NAME="my_memory_store"
        
async def main() -> None:
    print('---IN main--')

    project = AIProjectClient(
        endpoint=os.environ["FONDRY_PROJECT_ENDPOINT"],        
        credential=DefaultAzureCredential(),                
    )
    openai = project.get_openai_client()

    ################################
    # Create memory store
    #
    options = MemoryStoreDefaultOptions(
        chat_summary_enabled=True,
        user_profile_enabled=True
    )
    definition = MemoryStoreDefaultDefinition(
        chat_model=CHAT_MODEL_NAME,
        embedding_model=EMBEDDING_MODEL_NAME,
        options=options
    )

    memory_store = project.beta.memory_stores.create(
        name=MEMORY_STORE_NAME,        
        definition=definition,
        description="Memory store for my agent"
    )
    print(f"Memory store: {memory_store.name}")

 
    ###################################
    # Conversation
    #
    scope = "user_123"
    
    tools = [
        {
            "type": "memory_search_preview",
            "memory_store_name": MEMORY_STORE_NAME,
            "scope": scope,
        }
    ]

    response = openai.responses.create(
        model=CHAT_MODEL_NAME,
        input="What is the largest city in France?",
        tools=tools,
    )
    print(response.output_text)

    ###################################
    # Delete memory store
    #
    project.beta.memory_stores.delete(
        name=MEMORY_STORE_NAME        
    )
    
if __name__ == "__main__":
    print('---------start------------')
    asyncio.run(main())
    

