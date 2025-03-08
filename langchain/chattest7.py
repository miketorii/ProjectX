import os
#from openai import AzureOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

print("----------start----------------")

llm = AzureChatOpenAI(
#    azure_deployment="MyModel",  # or your deployment
#    api_version="2024-02-01",  # or your api version
    azure_deployment="my-gpt-4o-1",
    api_version="2024-08-01-preview",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("こんにちは。私はジョンと言います"),
    AIMessage("こんにちは、ジョンさん。どのようにお手伝いできますか？"),
    HumanMessage("私の名前がわかりますか？"),    
]

ai_msg = llm.invoke(messages)

#print(ai_msg)
print(ai_msg.content)

print("----------end------------------")

