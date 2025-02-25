import os
from langchain_openai import AzureChatOpenAI

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from IPython.display import Image
from PIL import Image as PILImage
import io

print("----------start----------------")

class State(TypedDict):
    value: str

def node(state: State, config: RunnableConfig):
    return {"value": "1"}

def node2(state: State, config: RunnableConfig):
    return {"value": "2"}

graph_builder = StateGraph(State)

graph_builder.add_node("node", node)
graph_builder.add_node("node2", node2)

graph_builder.add_edge("node", "node2")

graph_builder.set_entry_point("node")
graph_builder.set_finish_point("node2")

graph = graph_builder.compile()

pngdata = graph.get_graph().draw_mermaid_png()
image = PILImage.open(io.BytesIO(pngdata))
image.save("graph.png")

print("----------end------------------")





