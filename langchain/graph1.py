from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage

from langgraph.graph import StateGraph
from typing_extensions import Annotated
from langgraph.graph.message import add_messages

###############################################
#
#
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    extra_field: int

###############################################
#
#    
def node(state: State):
    messages = state["messages"]
    new_message = AIMessage("Hello!")

    return {"messages": messages + [new_message], "extra_field": 10}

###############################################
#
#
def process1():
    print("-----------start--------------")

    graph_builder = StateGraph(State)
    graph_builder.add_node(node)
    graph_builder.set_entry_point("node")
    graph = graph_builder.compile()

    result = graph.invoke({"messages": [HumanMessage("Hi")]})

    print(result)

    for message in result["messages"]:
        message.pretty_print()

    print("------------end---------------")

###############################################
#
#        
def process2():
    print("------------process2------------")
    graph = StateGraph(State).add_node(node).set_entry_point("node").compile()

    input_message = {"role":"user", "content":"HiHi"}
    result = graph.invoke({"messages":[input_message]})

    for message in result["messages"]:
        message.pretty_print()

    print("------------end---------------")
    
###############################################
#
#    
def process3():
    print("process3")    
    
###############################################
#
#    
def process(num: int):
    if num == 1:
        process1()
    elif num == 2:
        process2()
    else:
        process3()
        
###############################################
#
#
if __name__ == "__main__":
    process(2)
    
    
    
