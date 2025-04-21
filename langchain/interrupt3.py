from langchain_openai import AzureChatOpenAI

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage

################################################
#
#
@tool
def play_song_on_spotify(song: str):
    """Play a song on Spotify"""
    return f"Successfully played {song} on Spotify!"

@tool
def play_song_on_apple(song: str):
    """Play a song on Apple Music"""
    return f"Successfully played {song} on Apple Music!"

################################################
#
#
class AgentState():
    def __init__(self, model: AzureChatOpenAI):
        self.model = model

    def _should_continue(self, state):
        print("-----should continue------")
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"
        
    def _call_model(self, state):
        print("-----call model------")
        messages = state["messages"]
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def process(self):
        print("----------start------------")

        tools = [play_song_on_apple, play_song_on_spotify]
        tool_node = ToolNode(tools)

        model = self.model.bind_tools(tools, parallel_tool_calls=False)

        ###########################################
        #
        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", self._call_model)
        workflow.add_node("action", tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "action",
                "end": END
            }
        )
        workflow.add_edge("action", "agent")

        memory = MemorySaver()

        app = workflow.compile(checkpointer=memory)
        
        config = {"configurable":{"thread_id":"1"}}

        input_message = HumanMessage(content="Can you play Taylor Swift's most popular song?")

        for event in app.stream({"messages":[input_message]}, config, stream_mode="values"):
            event["messages"][-1].pretty_print()
        
        print("----------end------------")        

################################################
#
#        
if __name__ == "__main__":
    
    model = AzureChatOpenAI(
        azure_deployment="my-gpt-4o-1",
        api_version="2024-08-01-preview",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    agent = AgentState(model)
    agent.process()
    
