import os
from langchain_openai import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

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

#messages = [
#    (
#        "system",
#        "",
#    ),
#    ("human", "将来性のある日本のAIスタートアップは？"),
#]

template = """
You are a creative consultant brainstorming names for businesses.

You must follow the following principles:
{principles}

Please generate a numerical list of five catchy names for a start-up
in the {industry} industry that deals with {context}?

Here is an example of the format:
1. Name1
2. Name2
3. Name3
4. Name4
5. Name5
"""

system_prompt = SystemMessagePromptTemplate.from_template(template)
chat_prompt = ChatPromptTemplate.from_messages([system_prompt])

chain = chat_prompt | llm

result = chain.invoke({
    "industry" : "医療",
    "context":'''患者の記録を自動的に要約するAIソリューションの作成''',
    "principles":'''1. 名前は短く覚えやすいものにすること。
                   2. 名前は発音しやすいものにすること。
                   3. 名前は独自のもので、他の企業がすでに使用していないものにすること。'''
})

print(result)
print(result.content)

print("-------------------------------")


print("----------end------------------")





