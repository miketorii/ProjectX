import operator
from typing import Annotated, Any, Sequence

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    which: str

def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate":["A"]}

def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate":["B"]}

def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate":["C"]}

def d(state: State):
    print(f'Adding "D" to {state["aggregate"]}')
    return {"aggregate":["D"]}

def b_2(state: State):
    print(f'Adding "B_2" to {state["aggregate"]}')
    return {"aggregate":["B_2"]}    

def e(state: State):
    print(f'Adding "E" to {state["aggregate"]}')
    return {"aggregate":["E"]}    

def process1():
    builder = StateGraph(State)
    builder.add_node(a)
    builder.add_node(b)
    builder.add_node(b_2)
    builder.add_node(c)
    builder.add_node(d)

    builder.add_edge(START,"a")
    builder.add_edge("a","b")
    builder.add_edge("a","c")

    #builder.add_edge("b","d")
    builder.add_edge("b","b_2")
    builder.add_edge(["b_2", "c"],"d")

    #builder.add_edge("c","d")
    builder.add_edge("d",END)

    graph = builder.compile()
    result = graph.invoke({"aggregate":["FF"]}, {"configurable":{"thread_id":"foo"}})

    print(result)

def route_bc_or_cd(state: State) -> Sequence[str]:
    if state["which"] == "cd":
        return ["c","d"]
    return ["b","c"]
    
def process2():
    builder = StateGraph(State)
    builder.add_node(a)
    builder.add_node(b)
    builder.add_node(c)
    builder.add_node(d)
    builder.add_node(e)    

    builder.add_edge(START,"a")

    intermediates = ["b","c","d"]
    builder.add_conditional_edges(
        "a",
        route_bc_or_cd,
        intermediates
    )

    for node in intermediates:
        builder.add_edge(node,"e")
        
    builder.add_edge("e",END)

    graph = builder.compile()
#    result = graph.invoke({"aggregate":["FF"], "which":"bc"})
    result = graph.invoke({"aggregate":["FF"], "which":"cd"})    

    print(result)
    

if __name__ == "__main__":
    print("-------------Start------------")

    #process1()
    process2()    
    
    print("--------------End-------------")
