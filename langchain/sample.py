# main_supervisor.py
# ## 1.環境設定
import os
from dotenv import load_dotenv

load_dotenv("./.env")


# ## 2.ツール設定
from langchain_experimental.tools import PythonREPLTool

python_repl_tool = PythonREPLTool(verbose=True)


# ## 3.エージェント設定
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI

'''
llm = AzureChatOpenAI(
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("azure_endpoint"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
    streaming=True,
)
'''

llm = AzureChatOpenAI(
    azure_deployment="MyModel",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0.7,
    streaming=True,
#    temperature=0,
#    max_tokens=None,
#    timeout=None,
#    max_retries=2,
    # other params...
)

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


# ## 4.スーパーバイザー設定
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

members = ["sales_staff", "sales_manager", "sales_director"]
system_prompt = (
    " You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members
# Using Azure openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))


supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
#    | llm.bind_tools(tools=[function_def], tool_choice="route")
    | JsonOutputFunctionsParser()
)


# ## 5.グラフ設定
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END


# エージェントの状態は、各ノードへの入力。
class AgentState(TypedDict):
    # このアノテーションは、新しいメッセージが、
    # 常に現在のステートに追加されることをグラフに伝える。

    messages: Annotated[Sequence[BaseMessage], operator.add]
    # 'next'は、次のルーティング先を示す。
    next: str


# セールススタッフ　役割: 日々の顧客対応や商品の提案、売上データの入力などを行う。
sales_staff = create_agent(
    llm,
    [python_repl_tool],
    system_prompt="(階層1) 顧客対応と製品、サービス提案を担当。顧客からの質問に答え、適切な製品、サービスを推薦し、商談データ、売上予定データをシステムに記録します。",
)
sales_staff_node = functools.partial(agent_node, agent=sales_staff, name="Sales_Staff")

# セールスマネージャ　役割: セールスチームの管理と指導、販売戦略の立案、目標達成のためのリソース配分などを行う。
sales_manager = create_agent(
    llm,
    [python_repl_tool],
    system_prompt="(階層2) チームの管理と指導を担当。販売目標の設定、販売戦略の策定、パフォーマンスの監視、そしてチームメンバーへのフィードバック提供を行います。",
)
sales_manager_node = functools.partial(
    agent_node, agent=sales_manager, name="Sales_Manager"
)

# セールスディレクタ　役割: 全体のセールス戦略と目標の設定、事業部門の収益性分析、市場動向の監視、そして高いレベルでの顧客との関係構築を行う。
sales_director = create_agent(
    llm,
    [python_repl_tool],
    system_prompt="(階層3) 全体的な販売戦略と目標を設定し、セールス部門の収益性を分析。市場動向を監視し、重要な顧客との関係を築き、事業の成長を促進する。",
)
sales_director_node = functools.partial(
    agent_node, agent=sales_director, name="Sales_Director"
)

# ワークフロー設定
workflow = StateGraph(AgentState)
workflow.add_node("sales_staff", sales_staff_node)
workflow.add_node("sales_manager", sales_manager_node)
workflow.add_node("sales_director", sales_director_node)

# スーパーバイザーのワークフロー設定
workflow.add_node("supervisor", supervisor_chain)


# ## 6.ワークフロー構築（スーパーバイザーと各エージェント（グラフ）をエッジで連結）
for member in members:
    # 各メンバーはタスク終了後、スーパーバイザーに報告
    workflow.add_edge(member, "supervisor")
# スーパーバイザーは、ノードにルーティングするグラフステート"next"を入力
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# エントリーポイント（ここからスタート）
workflow.set_entry_point("supervisor")

# フローを動かすための準備
graph = workflow.compile()


# ## 7.スーパーバイザーを軸として各エージェント（ノード）と会話
for s in graph.stream(
    {
        "messages": [
            HumanMessage(
                content="このコードに事前に用意された各階層のagentを利用し、AIドリブンな世界における商品販売戦略について、agent間でブレストして下さい。" 
                "このコードに用意された各階層の1つのagentの発言は最大3回までです。発言は1回、100文字以内にして下さい。"             
            )
        ],
    },
    # グラフ内の最大ステップ数
    {"recursion_limit": 20},
):

    # 各エージェントの出力を個別に表示
    for key in ["sales_staff", "sales_manager", "sales_director"]:
        if key in s:
            # エージェント名でヘッダを出力
            # print(f"\n### {key} says:")
            messages = s[key]["messages"]
            for msg in messages:
                # エージェントからのメッセージ内容を出力
                print(msg.content)
                print("----\n")  # セクションの終わり



