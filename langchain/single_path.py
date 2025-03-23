import operator
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
class DecomposedTasks(BaseModel):
    values: list[str] = Field(
        default_factory=list,
#        min_items=3,
#        max_items=5,
        description="3-5個に分解されたタスク",
    )
    
################################################
#
#
class SinglePathPlanGenerationState(BaseModel):
    query: str = Field(..., description="ユーザーが入力したクエリ")
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(default="", description="最適化されたレスポンス定義")
    tasks: list[str] = Field(default_factory=list, description="実行するタスクのリスト")
    current_task_index: int = Field(default=0, description="現在実行中のタスクの番号")
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="実行済みタスクの結果リスト"
    )
    final_output: str = Field(default="", description="最終的な出力結果")
    
################################################
#
#
class QueryDecomposer:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, query: str) -> DecomposedTasks:
        print("----run in QueryDecomposer----")
        prompt = ChatPromptTemplate.from_template(
            f"CURRENT_DATE: {self.current_date}\n"
            "----------\n"
            "タスク: 与えられた目標を具体的で実行可能なタスクに分解してください。\n"
            "要件:\n"
            "1. 以下の行動だけで目標を達成すること。決して指定された以外の行動をとらないこと。\n"
            " - インターネットを利用して、目標を達成するための調査を行う。\n"
            "2. 各タスクは具体的かつ詳細に記載されており、単独で実行並びに検証可能な情報を含めること。一切抽象的な表現を含まないこと。\n"
            "3. タスクは実行可能な順序でリスト化すること。\n"
            "4. タスクは日本語で出力すること。\n"
            "目標: {query}"            
        )
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        print("----invoke in QueryDecomposer----")
        print(query)
        #return chain.invoke({"query": query})
        response = chain.invoke({"query": query})
        print(response)
        print("----done invoke in QueryDecomposer----")        
        return response
        
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
        graph.add_node("decompose_query", self._decompose_query)
        graph.add_node("execute_task", self._execute_task)
        graph.add_node("aggregate_results", self._aggregate_results)
        
        graph.set_entry_point("goal_setting")

        graph.add_edge("goal_setting","decompose_query")
        graph.add_edge("decompose_query", "execute_task")

        graph.add_conditional_edges(
            "execute_task",
            lambda state: state.current_task_index < len(state.tasks),
            {True: "execute_task", False: "aggregate_results"}
        )
        graph.add_edge("aggregate_results", END)
        
        return graph.compile()

    def _goal_setting(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        goal: Goal = self.passive_goal_creator.run(query=state.query)
        optimized_goal: OptimizedGoal = self.prompt_optimizer.run(query=goal.text)
        
        optimized_response: str = self.response_optimizer.run(query=optimized_goal.text)
        
        return { "optimized_goal": optimized_goal.text,
                 "optimized_response": optimized_response
        }

    def _decompose_query(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        decomposed_tasks: DecomposedTasks = self.query_decomposer.run(query=state.optimized_goal)
        return {"tasks": decomposed_tasks.values}
        
    def _execute_task(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        return {"results": null, "current_task_index": 1}
    
    def _aggregate_results(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        final_output = null
        return {"final_output": final_output}
        
    def run(self, query: str) -> str:
        print("-------initial state of SinglePathPlanGenerationState----")
        initial_state = SinglePathPlanGenerationState(query=query)
        print(initial_state)
        print("---------------------------------------------------------")
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 100})
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
    

