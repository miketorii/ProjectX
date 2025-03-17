import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain

from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.runnables import RunnablePassthrough

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

prompt = ChatPromptTemplate.from_template(
'''
以下の文脈だけを踏まえて質問に回答してください。

文脈:"""
{context}
"""

質問:{question}

'''
)

retriever = TavilySearchAPIRetriever(k=3)

chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | output_parser

# invoke
ai_msg = chain.invoke("東京の今日の天気は？")
print(ai_msg)

print("----------end------------------")

