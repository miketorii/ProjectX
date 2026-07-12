import asyncio
import os
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

from azure.ai.projects.models import MemoryStoreDefaultDefinition, MemoryStoreDefaultOptions

from azure.ai.projects.models import MemorySearchOptions
from azure.core.credentials import AzureKeyCredential

load_dotenv()

FONDRY_MODE=os.getenv('FONDRY_NAME')
FONDRY_PROJECT_ENDPOINT=os.getenv("FONDRY_PROJECT_ENDPOINT")
AGENT_NAME=os.getenv("AGENT_NAME2")

CHAT_MODEL_NAME=os.getenv("CHAT_MODEL_NAME")
EMBEDDING_MODEL_NAME=os.getenv("EMBEDDING_MODEL_NAME")

FONDRY_PROJECT_KEY = os.environ["FONDRY_PROJECT_KEY"]

MEMORY_STORE_NAME="my_memory_store"
        
async def main() -> None:
    print('---IN main--')

#    print(FONDRY_PROJECT_KEY)
    
    project = AIProjectClient(
        endpoint=os.environ["FONDRY_PROJECT_ENDPOINT"],
#        credential=AzureKeyCredential(FONDRY_PROJECT_KEY),
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

    project.beta.memory_stores.delete(
        name=MEMORY_STORE_NAME        
    )
    
    memory_store = project.beta.memory_stores.create(
        name=MEMORY_STORE_NAME,        
        definition=definition,
        description="Memory store for my agent"
    )
    print(f"Memory store: {memory_store.name}")

    ##########################################################
    # 
    #
    scope = "user_123"

#    project = AIProjectClient(
#        endpoint=os.environ["FONDRY_PROJECT_ENDPOINT"],
#        credential=AzureKeyCredential(FONDRY_PROJECT_KEY),
#        credential=DefaultAzureCredential(),
#    )
    
    query_message = {"role":"user", "content": "What are my coffee preferences?", "type":"message"}

    print("----------------search_memories-------------------")
    search_response = project.beta.memory_stores.search_memories(
        name=MEMORY_STORE_NAME,
        scope=scope,
        items=[query_message],
        options=MemorySearchOptions(max_memories=5)
    )
    print(f"Found {len(search_response.memories)} memories")

    for memory in search_response.memories:
        print(f"  - Memory ID: {memory.memory_item.memory_id}, Content: {memory.memory_item.content}")
        

    ###################################
    # Delete memory store
    #
    project.beta.memory_stores.delete(
        name=MEMORY_STORE_NAME        
    )
    
if __name__ == "__main__":
    print('---------start------------')
    asyncio.run(main())
    
'''
    user_message = {
        "role": "user",
        "content": "I prefer dark roast coffee and usually drink it in the morning",
        "type": "message"
    }

    update_poller = project_client.beta.memory_stores.begin_update_memories(
        name=memory_store_name,
        scope=scope,
        items=[user_message], # Pass conversation items that you want to add to memory
        update_delay=0, # Trigger update immediately without waiting for inactivity
    )

    # Wait for the update operation to complete, but can also fire and forget
    update_result = update_poller.result()
    print(f"Updated with {len(update_result.memory_operations)} memory operations")
    for operation in update_result.memory_operations:
        print(
            f"  - Operation: {operation.kind}, Memory ID: {operation.memory_item.memory_id}, Content: {operation.memory_item.content}"
        )
    
'''
