import operator
from datetime import datetime
from typing import Annotated, Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from response_optimizer import Goal, PassiveGoalCreator
from response_optimizer import OptimizedGoal, PromptOptimizer, ResponseOptimizer

################################################
#
#
class TaskOption(BaseModel):
    description: str = Field(default="", description="タスクオプションの説明")
    
################################################
#
#
class Task(BaseModel):
    task_name: str = Field(..., description="タスクの名前")
    options: list[TaskOption] = Field(
        default_factory=list,
        description="2-3個のタスクオプション",
    )


################################################
#
#
class DecomposedTasks(BaseModel):
    values: list[Task] = Field(
        default_factory=list,
#        min_items=3,
#        max_items=5,
        description="3-5個に分解されたタスク",
    )
    
################################################
#
#
class MultiPathPlanGenerationState(BaseModel):
    query: str = Field(..., description="ユーザーが入力したクエリ")
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(default="", description="最適化されたレスポンス")    
    tasks: DecomposedTasks = Field(
        default_factory=DecomposedTasks,
        description="複数のオプションを持つタスクのリスト"
    )    
    current_task_index: int = Field(default=0, description="現在のタスクのインデックス")
    chosen_options: Annotated[list[int], operator.add] = Field(
        default_factory=list,
        description="各タスクで選択されたオプションのインデックス"
    )
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="実行されたタスクの結果"
    )
    final_output: str = Field(default="", description="最終出力")
    
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
            "タスク: 与えられた目標を3-5個の高レベルタスクに分解し、各タスクに2-3個の具体的なオプションを提供してください。\n"
            "要件:\n"
            "1. 以下の行動だけで目標を達成すること。決して指定された以外の行動をとらないこと。\n"
            " - インターネットを利用して、目標を達成するための調査を行う。\n"
            "2. 各高レベルタスクは具体的かつ詳細に記載されており、単独で実行並びに検証可能な情報を含めること。一切抽象的な表現を含まないこと。\n"
            "3. 各高レベルタスクに2-3個のアプローチまたはオプションを提供すること。\n"
            "4. タスクは時効可能な順序でリスト化すること。\n"
            "5. タスクは日本語で出力すること。\n\n"
            "REMEMBER: 実行できないタスク、ならびに選択肢は絶対に作成しないでください。\n\n"
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
class OptionPresenter:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm.configurable_fields(
            max_tokens=ConfigurableField(id="max_tokens")
        )

    def run(self, task: Task) -> int:
        task_name = task.task_name
        options = task.options

        print(f"\nタスク: {task_name}")
        for i, option in enumerate(options):
            print(f"{i+1}. {option.description}")

        choice_prompt = ChatPromptTemplate.from_template(
            "タスク: 与えられたタスクとオプションに基づいて、最適なオプションを選択してください。必ず番号のみで回答してください。\n\n"
            "なお、あなたは次の行動しかできません。\n"
            "- インターネットを利用して、目標を達成するための調査を行う。\n\n"
            "タスク: {task_name}\n"
            "オプション:\n {options_text}\n"
            "選択 (1-{num_options})"            
        )

        option_text = "\n".join(
            f"{i+1}. {option.description}" for i, option in enumerate(options)
        )
        
        chain = (
            choice_prompt
            | self.llm.with_config(configurable=dict(max_tokens=1))
            | StrOutputParser()
        )
        
        choice_str = chain.invoke(
            {
                "task_name": task_name,
                "options_text": options_text,
                "num_options": len(options),                
            }
        )
        print(f"==> エージェントの選択: {choice_str}")

        return int(choice_str.strip())-1
    
################################################
#
#
class TaskExecutor:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.tools = [TavilySearchResults(max_results=3)]

    def run(self, task: str) -> str:
        print("---run in TaskExecutor---")
        agent = create_react_agent(self.llm, self.tools)
        result = agent.invoke(
            {
                "messages": [
                    (
                        "human",
                        (
                            "以下のタスクを実行し、詳細な回答を提供してください。\n\n"
                            f"タスク: {task.task_name} \n\n"
                            "要件:\n"
                            "1. 必要に応じて提供されたツールを使用してください。\n"
                            "2. 実行は徹底的かつ包括的に行ってください。\n"
                            "3. 可能な限り具体的な事実やデータを提供してください。\n"
                            "4. 発見した内容を明確に要約してください。\n"
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
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm

    def run(self, query: str, response_definition: str, tasks: list[Task], chosen_options: list[int], results: list[str]) -> str:
        print("------run in ResultAggregator----")
        prompt = ChatPromptTemplate.from_template(
            "与えられた目標\n{query}\n\n"
            "調査結果:{task_results}\n\n"
            "与えられた目標に対し、調査結果を用いて、以下の指示に基づいてレスポンスを生成してください。\n"
            "{response_definition}"
        )
        
        task_results = self._format_task_results(tasks, chosen_options, results)
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke(
            {
                "query": query,
                "task_results": task_results,
                "response_definition": response_definition,
            }
        )
        
        print(response)
        print("------done run in ResultAggregator----")        
        return response

    @staticmethod
    def _format_task_results( tasks: list[Task], chosen_options: list[int], results: list[str] ) -> str:
        task_results = ""
        
        for i, (task, chosen_option, result) in enumerate( zip(tasks, chosen_options, results) ):
            task_name = task.task_name
            chosen_option_desc = task.options[chosen_option].description
            task_results += f"タスク {i+1}: {task_name}\n"
            task_results += f"選択されたアプローチ: {chosen_option_desc}\n"
            task_results += f"結果: {result}\n\n"
            
        return task_results
    
################################################
#
#
class MultiPathPlanGeneration:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.passive_goal_creator = PassiveGoalCreator(llm=llm)
        self.prompt_optimizer = PromptOptimizer(llm=llm)
        self.response_optimizer = ResponseOptimizer(llm=llm)
        self.query_decomposer = QueryDecomposer(llm=llm)

        self.option_presenter = OptionPresenter(llm=llm)
        
        self.task_executor = TaskExecutor(llm=llm)
        self.result_aggregator = ResultAggregator(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        graph = StateGraph(MultiPathPlanGenerationState)
        graph.add_node("goal_setting", self._goal_setting)
        graph.add_node("decompose_query", self._decompose_query)
        graph.add_node("present_options", self._present_options)
        graph.add_node("execute_task", self._execute_task)
        graph.add_node("aggregate_results", self._aggregate_results)
        
        graph.set_entry_point("goal_setting")

        graph.add_edge("goal_setting","decompose_query")
        graph.add_edge("decompose_query", "present_options") 
        graph.add_edge("present_options", "execute_task")

        graph.add_conditional_edges(
            "execute_task",
            lambda state: state.current_task_index < len(state.tasks.values),
            {True: "present_options", False: "aggregate_results"}
        )
        graph.add_edge("aggregate_results", END)
        
        return graph.compile()

    def _goal_setting(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        goal: Goal = self.passive_goal_creator.run(query=state.query)
        optimized_goal: OptimizedGoal = self.prompt_optimizer.run(query=goal.text)
        
        optimized_response: str = self.response_optimizer.run(query=optimized_goal.text)
        
        return { "optimized_goal": optimized_goal.text,
                 "optimized_response": optimized_response
        }

    def _decompose_query(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        tasks = self.query_decomposer.run(query=state.optimized_goal)
        return {"tasks": tasks}

    def _present_options(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        current_task = state.tasks.values[state.current_task_index]
        chosen_option = self.option_presenter.run(task=current_task)    
        return {"chosen_options", [chosen_option]}
    
    def _execute_task(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        current_task = state.tasks.values[state.current_task_index]
        chosen_option = current_task.options[state.chosen_options[-1]]
        result = self.task_executor.run(
            task=current_task,
            chosen_option=chosen_option,
        )
        return {"results": [result], "current_task_index": state.current_task_index + 1}
    
    def _aggregate_results(self, state: MultiPathPlanGenerationState) -> dict[str, Any]:
        final_output = self.result_aggregator.run(
            query=state.optimized_goal,
            response_definition=state.optimized_response,
            tasks=state.tasks.values,
            chosen_options=state.chosen_options,
            results=state.results,
        )
        return {"final_output": final_output}

    def run(self, query: str) -> str:
        print("-------initial state of MultiPathPlanGenerationState----")
        initial_state = MultiPathPlanGenerationState(query=query)
        print(initial_state)
        print("---------------------------------------------------------")
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 100})
        return final_state.get("final_output", "最終的な回答の生成に失敗しました。")
       
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
    
    agent = MultiPathPlanGeneration(llm=llm)
    result = agent.run(argtask)

    print("----------Final Result------------------")    
    print(result)
    
    print("----------end------------------")
    
################################################
#
#
if __name__ == "__main__":
    main()
    

