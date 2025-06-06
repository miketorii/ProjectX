from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode

@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."

@tool
def get_coolest_cities():
    """Get a list of coolest cities"""
    return "nyc, sf"

tools = [get_weather, get_coolest_cities]
tool_node = ToolNode(tools)

message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name":"get_weather",
            "args":{"location":"sf"},
            "id":"tool_call_id",
            "type":"tool_call"
        }
    ],
)

response = tool_node.invoke({"messages":[message_with_single_tool_call]})
print(response)

