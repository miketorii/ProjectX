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

    models = project_client.get_openai_client(api_version="2024-10-21")
    response = models.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a security engineer."},
            {"role": "user", "content": "Write the trend about AI security"}
        ],
    )

    print(response.choices[0].message.content)

    print('--------End func_main--------')
    
if __name__ == '__main__':
    print('-----Start-----')
    func_main()
    print('-----End-----')    
