import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# invoke
ai_msg = chain.invoke({"dish":"カレー"})
print(ai_msg)

# stream
#for chunk in chain.stream({"dish":"カレー"}):
#    print(chunk, end="", flush=True)

# batch
#ai_msgs = chain.batch([{"dish":"カレー"},{"dish":"うどん"}])
#print(ai_msgs[0])
#print(ai_msgs[1])

print("----------end------------------")

