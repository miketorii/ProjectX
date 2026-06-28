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

    project.beta.memory_stores.delete(
        name=MEMORY_STORE_NAME        
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
        input="Remember that my preferred seat is aisle",
        tools=tools,
    )
    print(response.output_text)

    for item in response.output:
        if getattr(item, "type", None) == "memory_command_call":
            print(item.type)
            print(item.arguments)
            print(item.status)

    forget_response = openai.responses.create(
        model=CHAT_MODEL_NAME,
        input="Forget my preferred seat.",
        tools=tools,
    )
    print(forget_response.output_text)

    for item in forget_response.output:
        if getattr(item, "type", None) == "memory_command_call":
            print(item.type)
            print(item.arguments)
            print(item.status)            

    scope = "user_123"

    user_message = {
        "role": "user",
        "content": "I prefer dark roast cofee and usually drink it in the morning",
        "type": "message"
    }

    update_poller = project.beta.memory_stores.begin_update_memories(
        name=MEMORY_STORE_NAME,
        scope=scope,
        items=[user_message],
        update_delay=0,
    )

    update_result = update_poller.result()
    print(f"Updated with {len(update_result.memory_operations)} memory operations")
    for operation in update_result.memory_operations:
        print(f" - Operation: {operation.kind}, Memory ID: {operation.memory_item.memory_id}, Content: {operation.memory_item.content}")

    new_message = {
        "role": "user",
        "content": "I also like cappuccinos in the afternoon",
        "type": "message"
    }

    new_update_poller = project.beta.memory_stores.begin_update_memories(
        name=MEMORY_STORE_NAME,
        scope=scope,
        items=[new_message],
        previous_update_id = update_poller.update_id,
        update_delay=0,        
    )
    new_update_result = new_update_poller.result()
    for operation in new_update_result.memory_operations:
        print(f" - Operation: {operation.kind}, Memory ID: {operation.memory_item.memory_id}, Content: {operation.memory_item.content}")
    
    
    ###################################
    # Delete memory store
    #
    project.beta.memory_stores.delete(
        name=MEMORY_STORE_NAME        
    )
    
if __name__ == "__main__":
    print('---------start------------')
    asyncio.run(main())
    

