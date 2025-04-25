from typing import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.types import StreamWriter
from langchain_openai import AzureChatOpenAI

import asyncio

class State(TypedDict):
    topic: str
    joke: str
    poem: str

print("------------start--------------")

joke_model = AzureChatOpenAI(
    azure_deployment="my-gpt-4o-1",
    api_version="2024-08-01-preview",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    tags=["joke"]
)

poem_model = AzureChatOpenAI(
    azure_deployment="my-gpt-4o-1",
    api_version="2024-08-01-preview",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    tags=["poem"]
)

async def call_model(state, config):
    topic = state["topic"]
    print("Writing joke...")
    joke_response = await joke_model.ainvoke(
        [{"role":"user", "content":f"Write a joke about {topic}"}],
        config
    )
    
    print("\n\nWriting poem...")
    poem_response = await poem_model.ainvoke(
        [{"role":"user", "content":f"Write a short poem about {topic}"}],
        config
    )

    return {"joke": joke_response.content, "poem": poem_response.content}

graph = StateGraph(State)
graph.add_node(call_model)
graph.add_edge(START, "call_model")
graphcomp = graph.compile()

async def process_graph_stream():
    async for msg, metadata in graphcomp.astream(
            {"topic": "cats"},
            stream_mode="messages",
    ):
        if msg.content:
            print(msg.content, end="|", flush=True)

asyncio.run(process_graph_stream())

print("\n\n-------------end---------------")


