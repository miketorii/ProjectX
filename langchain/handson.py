import operator
from typing import Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph

from langchain_openai import AzureChatOpenAI

print("------------start---------------")

ROLES = {
    "1" : {
        "name" : "一般知識エキスパート",
        "description" : "幅広い分野の一般的な質問に答える",
        "details" : "幅広い分野の一般的な質問に対して、正確で分かりやすい回答を提供してください。"        
    },
    "2" : {
        "name" : "生成AI製品エキスパート",
        "description" : "生成AIや関連製品、技術に関する専門的な質問に答える",
        "details" : "生成AIや関連製品、技術に関する専門的な質問に対して、最新の情報と深い洞察を提供してください"        
    },
    "3" : {
        "name" : "カウンセラー",
        "description" : "個人的な悩みや心理的な問題に対してサポートを提供する",
        "details" : "個人的な悩みや心理的な問題に対して、共感的で支援的な回答を提供し、可能であれば適切なアドバイスも行ってください"        
    }    
}

#print( ROLES )

#################################################################
#
# State class
#
class State(BaseModel):
    query: str = Field( ..., description="ユーザーからの質問")
    current_role: str = Field(default="", description="選定された回答ロール")
    messages: Annotated[list[str], operator.add] = Field(default=[], description="回答履歴")
    current_judge: bool = Field(default=False, description="品質チェックの結果")
    judgement_reason: str = Field(default="", description="品質チェックの判定理由")
    
workflow = StateGraph(State)

#################################################################
#
# Create client for LLM
#
llm = AzureChatOpenAI(
    azure_deployment="MyModel",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0.7,
    streaming=True,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

print("------------end---------------")
