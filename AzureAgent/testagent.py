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
        
        thread = project_client.agents.threads.create()
        print(f'Created thread ID: {thread.id}')

        question = """Draw a graph for a line with a slope of 4
        and y-intercept of 9 and provide the file to me?"""

        message = project_client.agents.messages.create(
            thread_id = thread.id,
            role="user",
            content = question,
        )
        print(f"Created message ID: {message['id']}")

        run = project_client.agents.runs.create_and_process(
            thread_id = thread.id,
            agent_id = agent.id,
            additional_instructions="""Please address the user as Jane Doe.
            The user has a premium account.""",
        )
        print(f"Run finished with satus: {run.status}")

        if run.status == "failed":
            print(f"Run failed: {run.last_error}")

        messages = project_client.agents.messages.list(thread_id=thread.id)
        print(f"Messages: {messages}")

        for message in messages:
            print(f"Role: {message.role}, Content: {message.content}")
            for this_content in message.content:
                print(f"Content Type: {this_content.type}, Conetnt Data: {this_content}")
                
        
    print('--------End func_main--------')
    
if __name__ == '__main__':
    print('-----Start-----')
    func_main()
    print('-----End-----')    
