import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

cot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザの質問にステップバイステップで回答してください。"),
        ( "human", "{question}"),
    ]
)

cot_chain = cot_prompt | llm | output_parser

summarize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ステップバイステップで考えた回答から結論だけ抽出してください"),
        ( "human", "{text}"),
    ]
)

summarize_chain = summarize_prompt | llm | output_parser

cot_summarize_chain = cot_chain | summarize_chain

# invoke
ai_msg = cot_summarize_chain.invoke({"question":"10 + 2 * 3"})
print(ai_msg)

# stream
#for chunk in chain.stream({"dish":"カレー"}):
#    print(chunk, end="", flush=True)

# batch
#ai_msgs = chain.batch([{"dish":"カレー"},{"dish":"うどん"}])
#print(ai_msgs[0])
#print(ai_msgs[1])

print("----------end------------------")

