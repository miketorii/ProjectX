import operator
from datetime import datetime
from typing import Annotated, Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from response_optimizer import Goal, PassiveGoalCreator
from response_optimizer import OptimizedGoal, PromptOptimizer, ResponseOptimizer

from reflection_manager import Reflection,ReflectionManager, TaskReflector


################################################
#
#
def format_reflections(reflections: list[Reflection]) -> str:
    return (
        "\n\n".join(
            f"<ref_{i}><task>{r.task}</task><reflection>{r.reflection}</reflection></ref_{i}>"
            for i, r in enumerate(reflections)
        )
        if reflections
        else "No relevant past reflections"
    )

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
class ReflectiveAgentState(BaseModel):
    query: str = Field(..., description="ユーザーが最初に入力したクエリ")
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(default="", description="最適化されたレスポンス定義")    
    tasks: list[str] = Field(default_factory=list, description="実行するタスクのリスト")
    current_task_index: int = Field(default=0, description="現在実行中のタスクの番号")
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="実行済みタスクの結果リスト"
    )
    reflection_ids: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="リフレクション結果のIDリスト"
    )    
    final_output: str = Field(default="", description="最終的な出力結果")
    retry_count: int = Field(default=0, description="タスクの再試行回数")

################################################
#
#
class ReflectiveGoalCreator:
    def __init__(self, llm: AzureChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        self.passive_goal_creator = PassiveGoalCreator(llm=self.llm)
        self.prompt_optimizer = PromptOptimizer(llm=self.llm)

    def run(self, query: str) -> str:
        relevant_reflections = self.reflection_manager.get_relevant_reflections(query)
        reflection_text = format_reflections(relevant_reflections)

        query = f"{query}\n\n目標を設定する際に以下の過去の振り返りを考慮すること:\n{reflection_text}"
        goal: Goal = self.passive_goal_creator.run(query=query)
        optimized_goal: OptimizedGoal = self.prompt_optimizer.run(query=goal.text)
        
        return optimized_goal.text
    
################################################
#
#
class ReflectiveResponseOptimizer:
    def __init__(self, llm: AzureChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        self.response_optimizer = ResponseOptimizer(llm=llm)

    def run(self, query: str) -> str:

        optimized_response : str = ""
        
        return optimized_response

################################################
#
#
class QueryDecomposer:
    def __init__(self, llm: AzureChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm.with_structured_output(DecomposedTasks)
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.reflection_manager = reflection_manager

    def run(self, query: str) -> DecomposedTasks:
        print("----run in QueryDecomposer----")
        relevant_reflections = self.reflection_manager.get_relevant_reflections(query)
        reflection_text = format_reflections(relevant_reflections)
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
        chain = prompt | self.llm
        print("----invoke in QueryDecomposer----")
        print(query)
        #return chain.invoke({"query": query})
        tasks = chain.invoke({"query": query, "reflectikons": reflection_text})
        print(tasks)
        print("----done invoke in QueryDecomposer----")        
        return tasks


################################################
#
#
class TaskExecutor:
    def __init__(self, llm: AzureChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        self.current_date = datetime.now().strftime("%Y-%m-%d")        
        self.tools = [TavilySearchResults(max_results=3)]

    def run(self, task: str) -> str:
        print("---run in TaskExecutor---")
        relevant_reflections = self.reflection_manager.get_relevant_reflections(task)
        reflection_text = format_reflections(relevant_reflections)        
        agent = create_react_agent(self.llm, self.tools)
        result = agent.invoke(
            {
                "messages": [
                    (
                        "human",
                        (
                            "次のタスクを実行し、詳細な回答を提供してください。\n\n"
                            f"タスク: {task} \n\n"
                            "要件:\n"
                            "1. 必要に応じて提供されたツールを使用してください。\n"
                            "2. 実行は徹底的かつ包括的に行ってください。\n"
                            "3. 可能な限り具体的な事実やデータを提供してください。\n"
                            "4. 発見した内容を明確に要約してください。\n"
                            f"5. 以下の過去の振り返りを考慮すること:\n{reflection_text}\n"                            
                         ),
                     )
                ]
            }
        )
        print(result)
        print(result["messages"][-1].content)
        print("---done run in TaskExecutor---")        
        return result["messages"][-1].content


################################################
#
#
class ResultAggregator:
    def __init__(self, llm: AzureChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        self.current_date = datetime.now().strftime("%Y-%m-%d")        

    def run(self, query: str, 
            results: list[str],
            reflection_ids: list[str],
            response_definition: str,            
            ) -> str:
        print("------run in ResultAggregator----")
        relevant_reflections = [
            self.reflection_manager.get_reflection(rid) for rid in reflection_ids
        ]
        
        prompt = ChatPromptTemplate.from_template(
            "与えられた目標\n{query}\n\n"
            "調査結果:{results}\n\n"
            "与えられた目標に対し、調査結果を用いて、以下の指示に基づいてレスポンスを生成してください。\n"
            "{response_definition}\n\n"
            "過去の振り返りを考慮すること:\n{reflection_text}\n"
        )
        
        results_str = "\n\n".join(
            f"Info {i+1}:\n{result}" for i, result in enumerate(results)
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke(
            {
                "query": query,
                "results": results_str,
                "response_definition": response_definition,
                "reflection_text": format_reflections(relevant_reflections),
            }
        )
        
        print(response)
        print("------done run in ResultAggregator----")        
        return response


'''
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
        current_task = state.tasks[state.current_task_index]
        result = self.task_executor.run(task=current_task)
        return {"results": [result], "current_task_index": state.current_task_index + 1}
    
    def _aggregate_results(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        final_output = self.result_aggregator.run(
            query=state.optimized_goal,
            response_definition=state.optimized_response,
            results=state.results,
        )
        return {"final_output": final_output}
        
    def run(self, query: str) -> str:
        print("-------initial state of SinglePathPlanGenerationState----")
        initial_state = SinglePathPlanGenerationState(query=query)
        print(initial_state)
        print("---------------------------------------------------------")
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 100})
        return final_state.get("final_output", "Failed to generate a final response.")
'''

################################################
#
#
class ReflectiveAgent:
    def __init__(
            self,
            llm: AzureChatOpenAI,
            reflection_manager: ReflectionManager,
            task_reflector: TaskReflector,
            max_retries: int = 2,
    ):
        self.reflection_manager = reflection_manager
        self.task_reflector = task_reflector
        self.reflective_goal_creator = ReflectiveGoalCreator(
            llm=llm, reflection_manager=self.reflection_manager
        )
        self.reflective_response_optimizer = ReflectiveResponseOptimizer(
            llm=llm, reflection_manager=self.reflection_manager
        )
        self.query_decomposer = QueryDecomposer(
            llm=llm, reflection_manager=self.reflection_manager
        )
        self.task_executor = TaskExecutor(
            llm=llm, reflection_manager=self.reflection_manager            
        )
        self.result_aggregator = ResultAggregator(
            llm=llm, reflection_manager=self.reflection_manager            
        )
        self.max_retries = max_retries
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        graph = StateGraph(ReflectiveAgentState)

        graph.add_node("goal_setting", self._goal_setting)
        graph.add_node("decompose_query", self._decompose_query)
        graph.add_node("execute_task", self._execute_task)        
        graph.add_node("reflect_on_task", self._reflect_on_task)        
        graph.add_node("update_task_index", self._update_task_index)
        graph.add_node("aggregate_results", self._aggregate_results)

        graph.set_entry_point("goal_setting")

        graph.add_edge("goal_setting", "decompose_query")
        graph.add_edge("decompose_query", "execute_task")
        graph.add_edge("execute_task", "reflect_on_task")

        graph.add_conditional_edges(
            "reflect_on_task",
            self._should_retry_or_continue,
            {
                "retry": "execute_task",
                "continue": "update_task_index",
                "finish": "aggregate_results",                
            }
        )

        graph.add_edge("update_task_index", "execute_task")
        graph.add_edge("aggregate_results", END)
        
        return graph.compile()
        

    def _goal_setting(self, state: ReflectiveAgentState) -> dict[str, Any]:
        optimized_goal: str  = self.reflective_goal_creator.run(query=state.query)
        optimized_response: str = self.reflective_response_optimizer.run(
            query=optimized_goal
        )
        
        return {
            "optimized_goal": optimized_goal,
            "optimized_response": optimized_response
        }

    def _decompose_query(self, state: ReflectiveAgentState) -> dict[str, Any]:
        tasks: DecomposedTasks = self.query_decomposer.run(query=state.optimized_goal)
        return {"tasks": tasks.values}
        
        
    def _execute_task(self, state: ReflectiveAgentState) -> dict[str, Any]:
        current_task = state.tasks[state.current_task_index]
        result = self.task_executor.run(task=current_task)
        return {
            "results": [result],
            "current_task_index": state.current_task_index
        }
        
    def _reflect_on_task(self, state: ReflectiveAgentState) -> dict[str, Any]:
        current_task = state.tasks[state.current_task_index]
        current_result = state.results[-1]
        reflection = self.task_reflector.run(task=current_task, result=current_result)
        return {
            "reflection_ids": [reflection.id],
            "retry_count": (
                state.retry_count + 1 if reflection.judgement.needs_retry else 0
            )
        }

    def _should_retry_or_continue(self, state: ReflectiveAgentState) -> str:
        latest_reflection_id = state.reflection_ids[-1]
        latest_reflection = self.reflection_manager.get_reflection(latest_reflection_id)
        if (
                latest_reflection
                and latest_reflection.judgement.needs_retry
                and state.retry_count < self.max_retries
        ):
            return "retry"
        elif state.current_task_index < len(state.tasks) - 1:
            return "continue"
        else:
            return "finish"
        
    def _update_task_index(self, state: ReflectiveAgentState) -> dict[str, Any]:
        return {"current_task_index": state.current_task_index + 1}
        
    def _aggregate_results(self, state: ReflectiveAgentState) -> dict[str, Any]:
        final_output = self.result_aggregator.run(
            query=state.optimized_goal,
            results=state.results,
            reflection_ids=state.reflection_ids,
            response_definition=state.optimized_response
        )
        
        return {"final_output": final_output}
        
    def run(self, query: str) -> str:
        print("-------initial state of ReflectiveAgentState----")
        initial_state = ReflectiveAgentState(query=query)
        print(initial_state)
        print("---------------------------------------------------------")
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 100})
        return final_state.get("final_output", "エラー: 出力に失敗しました。")


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

    reflection_manager = ReflectionManager(file_path="tmp/self_reflection_db.json")
    task_reflector = TaskReflector(llm=llm, reflection_manager=reflection_manager)
    agent = ReflectiveAgent(
        llm=llm,
        reflection_manager=reflection_manager,
        task_reflector = task_reflector
    )
    result = agent.run(argtask)

    print("----------Final result-----------------------")
    print(result)
    
    print("----------end------------------")
    
################################################
#
#
if __name__ == "__main__":
    main()
    

