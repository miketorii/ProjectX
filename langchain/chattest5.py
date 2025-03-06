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
入力がAIに関係するか回答してください。
'''

user_prompt1 = '''
AIの進化はすごい
'''

user_prompt2 = '''
今日は良い天気だ
'''

user_prompt3 = '''
ChatGPTはとても便利だ
'''

assist_prompt_true = '''
true
'''

assist_prompt_false = '''
false
'''

def generate_recipe() -> str:
    response = client.chat.completions.create(
        #    model="MyModel",
        model="my-gpt-4o-1",
        messages=[
            {"role": "system", "content": system_prompt },
            {"role": "user", "content": user_prompt1 },
            {"role": "assistant", "content": assist_prompt_true },            
            {"role": "user", "content": user_prompt2 },
            {"role": "assistant", "content": assist_prompt_false },                        
            {"role": "user", "content": user_prompt3 }
        ]
    )
    return response.choices[0].message.content

recipe = generate_recipe()

print(recipe)

print("----------end------------------")

