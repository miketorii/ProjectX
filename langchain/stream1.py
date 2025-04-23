from typing import TypedDict
from langgraph.graph import StateGraph, START

class State(TypedDict):
    topic: str
    joke: str

def refine_topic(state: State):
    return {"topic": state["topic"]+" and cats"}

def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']} " }

graph = StateGraph(State)
graph.add_node(refine_topic)
graph.add_node(generate_joke)
graph.add_edge(START, "refine_topic")
graph.add_edge("refine_topic","generate_joke")
graphcomp = graph.compile()

for chunk in graphcomp.stream(
        {"topic" : "ice cream"},
        stream_mode="values"
):
    print(chunk)

