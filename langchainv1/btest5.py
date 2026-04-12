import os
from openai import AzureOpenAI

from langchain_openai import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal, Union


import pandas as pd
import requests
import io
from tqdm import tqdm

##################################
#
#
class EnrichedTransactionInformation(BaseModel):
    transaction_type: Union[
        Literal["購入","引き出し","預入","請求書払い","返金"], None
    ]
    transaction_category: Union[
        Literal["食品","娯楽","交通","公共料金","家賃","その他"], None
    ]

def remove_back_slashes(string):
    cleaned_string = string.replace("\\","")
    return cleaned_string

######################################
#
#    
print("----------start----------------")

url = "https://storage.googleapis.com/oreilly-content/transaction_data_with_expanded_descriptions.csv"

downloaded_file = requests.get(url)

df = pd.read_csv(io.StringIO(downloaded_file.text))[:5]
print( df.head() )

temperature = 0.0

model = AzureChatOpenAI(
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

system_prompt = """
あなたは銀行取引の分析の専門家で、単一の取引を分類します。
必ず取引の種類とカテゴリーを返し、Noneは返さないでください。
出力形式は、以下の通りです。
{format_instructions}
"""

user_prompt = """
取引の内容は、以下の通りです。
{transaction}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt,),
        ("user", user_prompt,),
    ]
)

output_parser = PydanticOutputParser(pydantic_object=EnrichedTransactionInformation)

chain = prompt | model | StrOutputParser() | remove_back_slashes | output_parser

transaction = df.iloc[0]["Transaction Description"]

result = chain.invoke(
    {
        "transaction": transaction,
        "format_instructions": output_parser.get_format_instructions(),
    }
)

print(result.transaction_type)
print(result.transaction_category)

print("-----------------------------------------")

results = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    transaction = row["Transaction Description"]
    try:
        result = chain.invoke(
            {
                "transaction": transaction,
                "format_instructions": output_parser.get_format_instructions(),
            }
        )
    except:
        result = EnrichedTransactionInformation(
            transaction_type=None,
            transaction_category=None
        )

    results.append(result)

transaction_types = []
transaction_categories = []

for result in results:
    transaction_types.append(result.transaction_type)
    transaction_categories.append(result.transaction_category)

df["my_transaction_type"] = transaction_types
df["my_transaction_category"] = transaction_categories

print(df.head())

print("-------------------------------")
print("----------end------------------")





