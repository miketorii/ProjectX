import os
from langchain_openai import AzureChatOpenAI

print("----------start----------------")

llm = AzureChatOpenAI(
    azure_deployment="MyModel",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

messages = [
    (
        "system",
        "",
    ),
    ("human", "将来性のある日本のAIスタートアップは？"),
]
ai_msg = llm.invoke(messages)

print(ai_msg)
print(ai_msg.content)

print("-------------------------------")


print("----------end------------------")





