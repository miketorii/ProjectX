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

llm = AzureChatOpenAI(
    azure_deployment="MyModel",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

class State(TypedDict):
    messages: str

graph_builder = StateGraph(State)

def chat_bot(state: State):
    if state["messages"]:
        return { "messages": llm.invoke(state["messages"]) }

    return { "messages": "o user input provided"}

graph_builder.add_node("chat_bot", chat_bot)

graph_builder.set_entry_point("chat_bot")
graph_builder.set_finish_point("chat_bot")

graph = graph_builder.compile()

pngdata = graph.get_graph().draw_mermaid_png()
image = PILImage.open(io.BytesIO(pngdata))
image.save("graph2.png")

prompt = "将来性のある日本のAIスタートアップは？"

response = graph.invoke({"messages": [prompt]}, debug=True)

print(response["messages"].content)

print("----------end------------------")





