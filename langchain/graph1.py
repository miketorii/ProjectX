from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage

from langgraph.graph import StateGraph

###############################################
#
#
class State(TypedDict):
    messages: list[AnyMessage]
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
if __name__ == "__main__":
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
    
    
