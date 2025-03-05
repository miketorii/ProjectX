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
ユーザが入力した料理のレシピを考えてください。

出力形式は以下のJSON形式にしてください。
"""
{
 "材料":["材料1", "材料2"],
 "手順":["手順1","手順2"]
}
"""
'''

prompt = '''
以下の料理のレシピを考えてください。

料理名:"""
{dish}
"""

'''

def generate_recipe(dish: str) -> str:
    response = client.chat.completions.create(
        #    model="MyModel",
        model="my-gpt-4o-1",
        messages=[
            {"role": "system", "content": system_prompt },
            {"role": "user", "content": "カレー"}
#            {"role": "user", "content": prompt.format(dish=dish)}                        
        ]
    )
    return response.choices[0].message.content

recipe = generate_recipe("カレー")

print(recipe)

print("----------end------------------")

