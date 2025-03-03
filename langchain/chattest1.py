import os
from openai import AzureOpenAI

print("----------start----------------")

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
#  api_version="2024-02-01"
  api_version="2024-08-01-preview"        
)

response = client.chat.completions.create(
#    model="MyModel",
    model="my-gpt-4o-1",
    messages=[
        {"role": "system", "content": "質問に100文字程度で答えてください"},
        {"role": "user", "content": "プロンプトエンジニアリングとは？"},
#        {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
#        {"role": "user", "content": "Do other Azure AI services support this too?"}
    ]
)

print(response.choices[0].message.content)


print("----------end------------------")
