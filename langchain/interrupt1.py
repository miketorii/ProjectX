import operator
from typing import Annotated, Any, Sequence, Literal

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    input: str
    user_feedback: str

def a(state: State):
    print(f'Node "A" sees {state["aggregate"]}')
    return {"aggregate":["A"]}

def step_1(state: State):
    print("---Step 1---")
    pass

def human_feedback(state: State):
    print("---human_feedback---")
    feedback = interrupt("Please provide feedback:")
    return {"user_feedback": feedback}

def step_3(state: State):
    print("---Step 3---")
    pass

def process1():
    builder = StateGraph(State)
    builder.add_node("step_1", step_1)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("step_3", step_3)        

    builder.add_edge(START,"step_1")
    builder.add_edge("step_1","human_feedback")
    builder.add_edge("human_feedback", "step_3")
    builder.add_edge("step_3", END)        

    memory = MemorySaver()
    
    graph = builder.compile(checkpointer=memory)

    initial_input = {"input":"hello mike"}
    thread = {"configurable": {"thread_id": "1"}}

    for event in graph.stream(initial_input, thread, stream_mode="updates"):
        print(event)
        print("\n")

    for event in graph.stream(
            Command(resume="go to step 3!"), thread, stream_mode="updates"
    ):
        print(event)
        print("\n")

    ret = graph.get_state(thread).values
    print(ret)
        
    
def process2():
    print("-----------process 2---------")

if __name__ == "__main__":
    print("-------------Start------------")

    process1()
    #process2()    
    
    print("--------------End-------------")
