import os
from openai import AzureOpenAI

print("----------start----------------")

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
#  api_version="2024-02-01"
  api_version="2024-08-01-preview"        
)

system_prompt = '''
ステップバイステップで考えてください。
'''

user_prompt1 = '''
10 + 2 * 3 - 4 * 2
'''

def generate_recipe() -> str:
    response = client.chat.completions.create(
        #    model="MyModel",
        model="my-gpt-4o-1",
        messages=[
            {"role": "system", "content": system_prompt },
            {"role": "user", "content": user_prompt1 },
        ]
    )
    return response.choices[0].message.content

recipe = generate_recipe()

print(recipe)

print("----------end------------------")

