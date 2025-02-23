from langchain.agents import load_tools
from langchain_experimental.tools import PythonREPLTool

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv("./.env")

#################################################################
#
#
def create_agent(llm, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),            
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    print(f"Executing {name} node!")
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}
    
#################################################################
#
#  Setup tools
#
python_repl_tool = PythonREPLTool(verbose=True)
#print(python_repl_tool)

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

#################################################################
#
# Create supervisor chain
#
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

members = ["sales_staff", "sales_manager", "sales_director"]

sysatem_prompt = (
    " You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

options = ["FINISH"] + members

function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next" : {
                "title": "Next",
                "anyOf": [
                    { "enum": options},
                ],
            }
        },
        "required": ["next"]
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sysatem_prompt),
        MessagesPlaceholder( variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members) )

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

#################################################################
#
# Create Agents
#
sales_staff = create_agent(
    llm,
    [python_repl_tool],
    system_prompt="(階層1) 顧客対応と製品、サービス提案を担当。顧客からの質問に答え、適切な製品、サービスを推薦し、商談データ、売上予定データをシステムに記録します。",
)

sales_manager = create_agent(
    llm,
    [python_repl_tool],
    system_prompt="(階層2) チームの管理と指導を担当。販売目標の設定、販売戦略の策定、パフォーマンスの監視、そしてチームメンバーへのフィードバック提供を行います。",    
)

sales_director = create_agent(
    llm,
    [python_repl_tool],
    system_prompt="(階層3) 全体的な販売戦略と目標を設定し、セールス部門の収益性を分析。市場動向を監視し、重要な顧客との関係を築き、事業の成長を促進する。",
)

#################################################################
#
# Create Nodes
#
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools

from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    
sales_staff_node = functools.partial( agent_node, agent=sales_staff, name="Sales_Staff")
sales_manager_node = functools.partial( agent_node, agent=sales_manager, name="Sales_Manager")
sales_director_node = functools.partial( agent_node, agent=sales_director, name="Sales_Director")

workflow = StateGraph(AgentState)
workflow.add_node("sales_staff", sales_staff_node)
workflow.add_node("sales_manager", sales_manager_node)
workflow.add_node("sales_director", sales_director_node)

workflow.add_node("supervisor", supervisor_chain)

#################################################################
#
# Create workflow
#
for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

workflow.set_entry_point("supervisor")

graph = workflow.compile()

#################################################################
#
# Execute task
#

for s in graph.stream(
    {
        "messages": [
            HumanMessage(
                content="このコードに事前に用意された各階層のagentを利用し、AIドリブンな世界における商品販売戦略について、agent間でブレストして下さい。" 
                "このコードに用意された各階層の1つのagentの発言は最大3回までです。発言は1回、100文字以内にして下さい。"             
            )
        ],
    },
    {"recursion_limit": 20},
):
    for key in ["sales_staff", "sales_manager", "sales_director"]:
        if key in s:
            # print(f"\n### {key} says:")
            messages = s[key]["messages"]
            for msg in messages:
                print(msg.content)
                print("----\n")  # セクションの終わり
'''
                
for s in graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="このコードに事前に用意された各階層のagentを利用し、AIドリブンな世界における商品販売戦略について、agent間でブレストして下さい。"
                    "このコードに用意された各階層の1つのagentの発言は最大3回までです。発言は1回、100文字以内にして下さい。"                       
                )
            ],
        },
        {"recursion_limit": 20},        
):
    for key in ["sales_staff", "sales_manager", "sales_director"]:
        if key in s:
            messages = s[key]["messages"]
            for msg in messages:
                print(msg.content)
                print("-----------\n")
'''
