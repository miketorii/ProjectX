import os
from dotenv import load_dotenv

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from azure.ai.agents.models import CodeInterpreterTool

load_dotenv()

MODEL_DEPLOYMENT_NAME=os.getenv('MODEL_DEPLOYMENT_NAME')
PROJECT_ENDPOINT=os.getenv("PROJECT_ENDPOINT")

def func_main():
    print('--------Start func_main--------')
    print(MODEL_DEPLOYMENT_NAME)
    print(PROJECT_ENDPOINT)

    project_client = AIProjectClient(
        endpoint = PROJECT_ENDPOINT,
        credential=DefaultAzureCredential()
    )

    with project_client:
        code_interpreter = CodeInterpreterTool()

        agent = project_client.agents.create_agent(
            model=MODEL_DEPLOYMENT_NAME,
            name="my-agent",
            instructions="""You politely help with math questions.
            Use the Code Interpreter tool when asked to visualize numbers.""",
            tools=code_interpreter.definitions,
            tool_resources=code_interpreter.resources
        )
        print(f'Created agent ID: {agent.id}')
        
    
    print('--------End func_main--------')
    
if __name__ == '__main__':
    print('-----Start-----')
    func_main()
    print('-----End-----')    
