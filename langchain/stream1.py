from typing import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.types import StreamWriter
from langchain_openai import AzureChatOpenAI

class State(TypedDict):
    topic: str
    joke: str

llm = AzureChatOpenAI(
    azure_deployment="my-gpt-4o-1",
    api_version="2024-08-01-preview",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def refine_topic(state: State):
    return {"topic": state["topic"]+" and cats"}

def generate_joke1(state: State):
    return {"joke": f"This is a joke about {state['topic']} " }

def generate_joke(state: State):
    response = llm.invoke(
        [
            {"role": "user", "content": f"Generate a joke about {state['topic']}"}
        ]
    )
    return {"joke": response.content }

def generate_joke3(state: State, writer: StreamWriter):
    writer({"custom_key": "Writing custom data while generating a joke"})
    return {"joke": f"This is a joke about {state['topic']} " }

graph = StateGraph(State)
graph.add_node(refine_topic)
graph.add_node(generate_joke)
graph.add_edge(START, "refine_topic")
graph.add_edge("refine_topic","generate_joke")
graphcomp = graph.compile()

for chunk in graphcomp.stream(
        {"topic" : "ice cream"},
#        stream_mode="values"
#        stream_mode="updates"
#        stream_mode="debug"
#        stream_mode="custom"
        stream_mode="messages"         
):
    print(chunk)

