from langchain_openai import AzureChatOpenAI

from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

################################################
#
#
class AskHuman(BaseModel):
    """Ask the human a question"""
    question: str
    
################################################
#
#
@tool
def search(query: str):
    """Call to surf the web"""
    print("-----------search---------------")
    print(query)
    return f"I looked up: {query}. Result: It's sunny in San Francisco, but you better look out if you're a Gemini."

################################################
#
#

################################################
#
#
class AgentState():
    def __init__(self, model):
        self.model = model

    def _call_model(self, state):
        print("--------call model-----------")
        messages = state["messages"]
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def _ask_human(self, state):
        print("--------ask human-----------")
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        ask = AskHuman.model_validate(state["messages"][-1].tool_calls[0]["args"])
        location = interrupt(ask.question)
        tool_message = [{"tool_call_id":tool_call_id, "type":"tool", "content":location}]
        return {"messages": tool_message}

    def _should_continue(self, state):
        print("--------should continue-----------")
        messages = state["messages"]
        last_message = messages[-1]

        if not last_message.tool_calls:
            return END
        elif last_message.tool_calls[0]["name"] == "AskHuman":
            return "ask_human"
        else:
            return "action"
        
    def run(self):
        print("----------start----------------")
        print("-----------run in AgentState--------------")
        
        tools = [search]
        tool_node = ToolNode(tools)
    
        self.model = self. model.bind_tools(tools + [AskHuman])

        print("----------Create graph-----------------------")
        workflow = StateGraph(MessagesState)
    
        workflow.add_node("agent", self._call_model)
        workflow.add_node("action", tool_node)
        workflow.add_node("ask_human", self._ask_human)
    
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self._should_continue)
        workflow.add_edge("action", "agent")
        workflow.add_edge("ask_human", "agent")    
    
        memory = MemorySaver()

        app = workflow.compile(checkpointer=memory)

        config = {"configurable":{"thread_id":"2"}}

        for event in app.stream(
                {
                    "messages": [
                        (
                        "user",
                        "Ask the user where they are, then look up the weather there"
                        )
                    ]
                },
                config,
                stream_mode="values"
        ):
            event["messages"][-1].pretty_print()

        ret = app.get_state(config).next
        print(ret)

        for event in app.stream(Command(resume="san francisco"), config, stream_mode="values" ):
            event["messages"][-1].pretty_print()            
        
        print("----------Final result-----------------------")
        print("----------end------------------")

################################################
#
#
        
################################################
#
#
if __name__ == "__main__":
    #main()

    model = AzureChatOpenAI(
        azure_deployment="my-gpt-4o-1",
        api_version="2024-08-01-preview",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    agent = AgentState(model)
    agent.run()
    

