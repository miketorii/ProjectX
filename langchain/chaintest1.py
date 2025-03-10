import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

print("----------start----------------")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザが入力した料理のレシピを考えてください"),  
        ( "human", "{dish}"),
    ]
)

llm = AzureChatOpenAI(
    azure_deployment="my-gpt-4o-1",
    api_version="2024-08-01-preview",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

chain = prompt | llm

ai_msg = chain.invoke({"dish":"カレー"})

print(ai_msg.content)

print("----------end------------------")

