import os
from openai import AzureOpenAI

from langchain_openai import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

##################################
#
#


class BusinessName(BaseModel):
    name: str = Field(description="事業名")
    rating_score: float = Field(description='''事業の評価スコア。0が最低評価で、10が最高評価。''')


class BusinessNames(BaseModel):
    names: List[BusinessName] = Field(description='''事業名のリスト''')

######################################
#
#    
print("----------start----------------")

temperature = 0.0

parser = PydanticOutputParser(pydantic_object=BusinessNames)

principles = """
原則：
- 名前は覚えやすいものでなければなりません。
- [industry]業界と会社の背景情報を使用して、効果的な名前を作成して下さい。
- 名前は発音しやすいものでなければなりません。
- 名前のみを返し、他のテキストや文字は含めないでください。
- 句読点や改行文字の\n、そのほかの記号の使用は避けて下さい。
- 名前は10文字以下でなければなりません。
"""

client = AzureChatOpenAI(
    azure_deployment="MyModel",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

#client = AzureOpenAI(
#    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
#    api_key=os.getenv("AZURE_OPENAI_KEY"),
#    api_version="2024-08-01-preview"
#)

template = """
テンプレート：
{industry}業界の新しいスタートアップ企業の事業名を５つ生成して下さい。
以下の原則に従う必要があります。
{principles}
{format_instructions}
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

prompt_and_model = chat_prompt | client

result = prompt_and_model.invoke(
    {
        "principles": principles,
        "industry": "宇宙産業",
        "format_instructions": parser.get_format_instructions(),
    }
)

print(parser.parse(result.content))


print("-------------------------------")
print("----------end------------------")





