from datetime import datetime
from typing import Annotated, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from response_optimizer import Goal, PassiveGoalCreator
from response_optimizer import OptimizedGoal, PromptOptimizer, ResponseOptimizer


################################################
#
#
class SinglePathPlanGenerationState(BaseModel):
    query: str = Field(..., description="ユーザーが入力したクエリ")
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(default="", description="最適化されたレスポンス定義")
    
################################################
#
#
class QueryDecomposer:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

################################################
#
#
class TaskExecutor:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm

################################################
#
#
class ResultAggregator:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm        
        
################################################
#
#
class SinglePathPlanGeneration:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.passive_goal_creator = PassiveGoalCreator(llm=llm)
        self.prompt_optimizer = PromptOptimizer(llm=llm)
        self.response_optimizer = ResponseOptimizer(llm=llm)
        
        self.query_decomposer = QueryDecomposer(llm=llm)
        self.task_executor = TaskExecutor(llm=llm)
        self.result_aggregator = ResultAggregator(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        graph = StateGraph(SinglePathPlanGenerationState)
        graph.add_node("goal_setting", self._goal_setting)
        graph.add_node("aggregate_results", self._aggregate_results)
        graph.set_entry_point("goal_setting")
        graph.add_edge("aggregate_results", END)
        return graph.compile()

    def _goal_setting(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        optimized_response = null
        return { "optimized_goal": null,
                 "optimized_response": optimized_response
        }
    
    def _aggregate_results(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        final_output = null
        return {"final_output": final_output}
        
    def run(self, query: str) -> str:
        return "run in SinglePathPlanGeneration"

################################################
#
#
def main():
    print("----------start----------------")
    
    llm = AzureChatOpenAI(
        azure_deployment="my-gpt-4o-1",
        api_version="2024-08-01-preview",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    argtask = "カレーライスの作り方"
    
    agent = SinglePathPlanGeneration(llm=llm)
    result = agent.run(argtask)
    print(result)
    
    print("----------end------------------")
    
################################################
#
#
if __name__ == "__main__":
    main()
    

