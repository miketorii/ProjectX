import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

print("----------start----------------")

llm = AzureChatOpenAI(
    azure_deployment="my-gpt-4o-1",
    api_version="2024-08-01-preview",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ( "human", "{input}"),
    ]
)

def upper(text: str) -> str:
    return text.upper()

chain = prompt | llm | output_parser | RunnableLambda(upper)

# invoke
ai_msg = chain.invoke({"input":"Hello!"})
print(ai_msg)

print("----------end------------------")

