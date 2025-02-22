from langchain.agents import load_tools
from langchain_experimental.tools import PythonREPLTool

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI

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
    

python_repl_tool = PythonREPLTool(verbose=True)
print(python_repl_tool)

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

sales_staff = create_agent(
    llm,
    [python_repl_tool],
    system_prompt="(階層1) 顧客対応と製品、サービス提案を担当。顧客からの質問に答え、適切な製品、サービスを推薦し、商談データ、売上予定データをシステムに記録します。",
)

