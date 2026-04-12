import os
from openai import AzureOpenAI

from langchain_openai import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

##################################
#
#
def most_frequent_classification(responses):
    count_dict = {}
    for classification in responses:
        count_dict[classification] = count_dict.get(classification, 0) + 1

    return max(count_dict, key=count_dict.get)


######################################
#
#
print("----------start----------------")

#client = AzureChatOpenAI(
#    azure_deployment="MyModel",  # or your deployment
#    api_version="2024-02-01",  # or your api version
#    temperature=0,
#    max_tokens=None,
#    timeout=None,
#    max_retries=2,
#    # other params...
#)

client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-08-01-preview"
)

base_template = """
以下の文を読み、「賞賛」「苦情」「中立」のいずれかに分類して下さい。
1.「太陽が輝いています」　- 中立
2.「あなたのサポートチームは素晴らしいです」 - 賞賛
3.「あなたのソフトウェアでひどい経験をしました」- 苦情

以下の原則に従わなければなりません。
- 単一の分類語のみを返して下さい。応答は、「賞賛」「苦情」「中立」の
いずれかでなければなりません。

- '''で囲まれたテキストの分類を行って下さい

'''{content}'''

分類：
"""

responses = []

for i in range(0,3):
    response = client.chat.completions.create(
        model="my-gpt-4o-1",
        messages=[
            {"role": "system", "content": base_template.format(content=
            '''外は雨ですが、私は素晴らしい一日を過ごしています。でも、
               人々がどうやって生きているのか理解できません。本当に悲しいです''')
             },
        ],
    )

    responses.append(response.choices[0].message.content.strip())


print(most_frequent_classification(responses))


print("-------------------------------")
print("----------end------------------")





