import operator
from typing import Annotated, Any, Sequence, Literal

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from langgraph.errors import GraphRecursionError

class State(TypedDict):
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Node "A" sees {state["aggregate"]}')
    return {"aggregate":["A"]}

def b(state: State):
    print(f'Node "B" sees {state["aggregate"]}')
    return {"aggregate":["B"]}

def c(state: State):
    print(f'Node "C" sees {state["aggregate"]}')
    return {"aggregate":["C"]}

def d(state: State):
    print(f'Node "D" sees {state["aggregate"]}')
    return {"aggregate":["D"]}

def route(state: State) -> Literal["b", END]:
    if len(state["aggregate"]) < 9:
        return "b"
    else:
        return END
    
def process1():
    builder = StateGraph(State)
    builder.add_node(a)
    builder.add_node(b)

    builder.add_edge(START,"a")
    builder.add_conditional_edges("a", route)
    builder.add_edge("b","a")

    graph = builder.compile()

    result = graph.invoke({"aggregate":["S"]})
    print(result)
    
'''    
    try:
        result = graph.invoke( {"aggregate":["S"]}, {"recursion_limit":4} )
        print(result)
    except GraphRecursionError:
        print("Recursion Error")
'''

def process2():
    print("-----------process 2---------")
    builder = StateGraph(State)
    builder.add_node(a)
    builder.add_node(b)
    builder.add_node(c)
    builder.add_node(d)    

    builder.add_edge(START,"a")
    builder.add_conditional_edges("a", route)
    builder.add_edge("b","c")
    builder.add_edge("b","d")
    builder.add_edge(["c","d"],"a")    

    graph = builder.compile()

    result = graph.invoke({"aggregate":["S"]})
    print(result)    

if __name__ == "__main__":
    print("-------------Start------------")

    #process1()
    process2()    
    
    print("--------------End-------------")
