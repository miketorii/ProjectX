import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I work in Japan. I live in Tokyo."),
]
ai_msg = llm.invoke(messages)

#print(ai_msg)
print(ai_msg.content)

print("-------------------------------")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}. Translate the user sentence."
        ),
        ("human", "{input}")
    ]
)

chain = prompt | llm
chain_msg = chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)

#print(chain_msg)
print(chain_msg.content)

print("----------end------------------")





