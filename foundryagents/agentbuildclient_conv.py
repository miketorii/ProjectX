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

    conversation = openai.conversations.create(
        items=[
            {
                "type": "message",
                "role": "user",
                "content": "What is the largest city in France?",                
            }
        ],
    )
    print(f"Conversation ID: {conversation.id}")

    openai.conversations.items.create(
        conversation_id=conversation.id,
        items=[
            {
                "type": "message",
                "role": "user",
                "content": "What about Germany?",
            }
        ],
    )    

if __name__ == "__main__":
    print('---------start------------')
    asyncio.run(main())
    

