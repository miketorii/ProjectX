import operator
from typing import Annotated

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from langchain_openai import AzureChatOpenAI

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import END

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
    #azure_deployment="MyModel",  # or your deployment
    #api_version="2024-02-01",  # or your api version
    azure_deployment="my-gpt-4o-1",  # or your deployment    
    api_version="2024-08-01-preview",  # or your api version
    temperature=0.7,
    streaming=True,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

#################################################################
#
# Node definition
#

#################################################################
#
# selection node
#
def selection_node(state: State) -> dict[str, Any]:
    print("------selection_node-------")    
    query = state.query
    role_options = "\n".join([ f"{k}. {v['name']}: {v['description']} " for k, v in ROLES.items() ])
    prompt = ChatPromptTemplate.from_template(
"""質問を分析し、最も適切な回答担当ロールを選択してください

選択肢:
{role_options}

回答は選択肢の番号(1、2、または3)のみを返してください。

質問: {query}
""".strip()

    )

    chain = prompt | llm.with_config( configurable=dict(max_tokens=1) ) | StrOutputParser()
    role_number = chain.invoke({"role_options": role_options, "query": query})

    selected_role = ROLES[role_number.strip()]["name"]
    return {"current_role" : selected_role}

#################################################################
#
# answering node
#
def answering_node(state: State) -> dict[str, Any]:
    print("------answering_node-------")
    query = state.query
    role = state.current_role
    role_details = "\n".join([ f"- {v['name']}: {v['details']} " for v in ROLES.values() ])
    prompt = ChatPromptTemplate.from_template(
"""あなたは{role}として回答してください。以下の質問に対して、あなたの役割に基づいた適切な回答を提供してください。

役割の詳細:
{role_details}

質問: {query}

回答:""".strip()
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"role": role, "role_details": role_details, "query": query})

    return {"messages": [answer]}

#################################################################
#
# check node
#
class Judgement(BaseModel):
    reason: str = Field( default="", description="判定理由")
    judge: bool = Field( default=False, description="判定結果")


def check_node(state: State) -> dict[str, Any]:
    print("------check_node-------")
    query = state.query
    answer = state.messages[-1]
    print(query)
    print("-------------------")
    print(answer)

    '''
    prompt = ChatPromptTemplate.from_template(
"""以下の回答の品質をチェックし、問題がある場合は'False'、問題がない場合は'True'を回答してください。また、その判断理由も説明してください。

ユーザーからの質問: {query}
回答: {answer}
""".strip()
    )
    print("---chain---")
    chain = prompt | llm.with_structured_output(Judgement)
    print("---invoke---")    
    result: Judgement = chain.invoke({"query": query, "answer": answer})

    print(result)
    
    return {
        "current_judge": result.judge,
        "judgement_reason": result.reason
    }
    '''

    print("---prompt---")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            'system',
            """以下の回答の品質をチェックし、問題がある場合は'False'、問題がない場合は'True'を回答してください。また、その判断理由も説明してください。

            ユーザーからの質問: {query}
            回答: {answer}
            """,
        ),
        ('placeholder', '{query}'),
        ('placeholder', '{answer}'),        
    ])
    
    print("---chain---")
    chain = prompt | llm.with_structured_output(Judgement)
    print("---invoke---")
    result: Judgement = chain.invoke({"messages": query})
    #result: Judgement = chain.invoke({"query": query, "answer": answer})        

    print(result)
    
    return {
        "current_judge": result.judge,
        "judgement_reason": result.reason
    }

#################################################################
#
# add nodes and edges
#
workflow.add_node("selection", selection_node)
workflow.add_node("answering", answering_node)
workflow.add_node("check", check_node)

workflow.set_entry_point("selection")

workflow.add_edge("selection","answering")
workflow.add_edge("answering","check")

# conditional edge
workflow.add_conditional_edges(
    "check",
    lambda state: state.current_judge,
    {True: END, False: "selection"}
)

#################################################################
#
# compile and execute
#
compiled = workflow.compile()

initial_state = State(query="生成AIについて教えてください")
result = compiled.invoke(initial_state)

print( result["messages"][-1] )

print("------------end---------------")

