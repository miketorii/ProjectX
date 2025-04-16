from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

print("------------start------------")

class State(TypedDict):
    value_1: str
    value_2: int
    
def step_1(state: State):
    return {"value_1": "a"}

def step_2(state: State):
    current_value_1 = state["value_1"]
    return {"value_1": f"{current_value_1} b"}

def step_3(state: State):
    return {"value_2": "10"}


graph_builder = StateGraph(State)

graph_builder.add_node(step_1)
graph_builder.add_node(step_2)
graph_builder.add_node(step_3)

graph_builder.add_edge(START, "step_1")
graph_builder.add_edge("step_1", "step_2")
graph_builder.add_edge("step_2", "step_3")

graph = graph_builder.compile()

result = graph.invoke({"value_1": "c"})

print(result)

print("------------end------------")
