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
        input="What happened in the news today?"
    )

    for item in response.output:
        if item.type == "web_search_call":
            print(f"[Tool] Web search: status={item.status}")
        elif item.type == "function_call":
            print(f"[Tool] Function call: {item.name}({item.arguments})")        
        elif item.type == "file_search_call":
            print(f"[Tool] File search: {item.status}")        
        elif item.type == "message":
            print(f"[Assistant] : {item.content[0].text})")        
        
if __name__ == "__main__":
    print('---------start------------')
    asyncio.run(main())
    

