from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

################################################
#
#
class Goal(BaseModel):
    description: str = Field(..., description="目標の説明")

    @property
    def text(self) -> str:
        return f"{self.description}"

################################################
#
#
class PassiveGoalCreator:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        
    def run(self, query: str) -> Goal:
        prompt = ChatPromptTemplate.from_template(
            "ユーザーの入力を分析し、明確で実行可能な目標を生成してください。\n"
            "要件:\n"
            "1. 目標は具体的かつ明確であり、実行可能なレベルで詳細化されている必要があります。\n"
            "2. あなたが実行可能な行動は以下の行動だけです。\n"
            "  - インターネットを利用して、目標を達成するための調査を行う。\n"
            "  - ユーザーのためのレポートを生成する。\n"
            "3. 決して2.以外の行動を取ってはいけません。\n"
            "ユーザーの入力: {query}"            
        )
        chain = prompt | self.llm.with_structured_output(Goal)
        return chain.invoke({"query":query})

################################################
#
#
class OptimizedGoal(BaseModel):
    description: str = Field(..., description="目標の説明")
    metrics: str = Field(..., description="目標の達成度を測定する方法")

    @property
    def text(self) -> str:
        return f"{self.description}(測定基準: {self.metrics})"
    
################################################
#
#
class PromptOptimizer:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm

    def run(self, query: str) -> OptimizedGoal:
        prompt = ChatPromptTemplate.from_template(
            "あなたは目標設定の専門家です。以下の目標をSMART原則(Specific: 具体的、Mesurable: 測定可能、Achievable: 達成可能、Relevant: 関連性が高い、Time-bound: 期限がある)に基づいて最適化してください。\n\n"
            "元の目標:\n"
            "{query}\n\n"
            "指示:\n"
            "1. 元の目標を分析し、不足している要素や改善点を特定してください。\n"
            "2. あなたが実行可能な行動は以下の行動だけです。\n"
            "  - インターネットを利用して、目標を達成するための調査を行う。\n"
            "  - ユーザーのためのレポートを生成する。\n"
            "3. SMART原則の各要素を考慮しながら、目標を具体的かつ詳細に記載してください。\n"
            "  - 一切抽象的な表現を含んではいけません。\n"
            "  - 必ず全ての単語が実行可能かつ具体的であることを確認してください。\n"            
            "4. 目標の達成度を測定する方法を具体的かつ詳細に記載してください。\n"
            "5. 元の目標で期限が指定されていない場合は、期限を考慮する必要はありません。\n"
            "6. REMEMBER: 決して2.以外の行動を取ってはいけません。\n"
        )

        chain = prompt | self.llm.with_structured_output(OptimizedGoal)
        return chain.invoke({"query":query})

################################################
#
#
class ResponseOptimizer:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm

    def run(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはAIエージェントシステムのレスポンス最適化スペシャリストです。与えられた目標に対して、エージェントが目標にあったレスポンスを返すためのレスポンス仕様を策定してください。",
                ),
                (
                    "human",
                    "以下の手順に従って、レスポンス最適化プロンプトを作成してください。\n\n"
                    "1. 目標分析\n"
                    "提示された目標を分析し、主要な要素や意図を特定してください。\n\n"
                    "2. レスポンス仕様の策定\n"
                    "目標達成のための最適なレスポンス仕様を考案してください。トーン、構造、内容の焦点などを考慮に入れてください。\n\n"
                    "3. 具体的な指示の作成\n"
                    "事前に収集された情報から、ユーザーの期待に沿ったレスポンスをするために必要な、AIエージェントに対する明確で実行可能な指示を作成してください。あなたの指示によってAIエージェントが実行可能なのは、既に調査済みの結果をまとめることだけです。インターネットへのアクセスはできません。\n\n"
                    "4. 例の提供\n"
                    "可能であれば、目標に沿ったレスポンスの例を１つ以上含めてください。\n\n"
                    "5. 評価基準の設定\n"
                    "レスポンスの効果を測定するための基準を定義してください。\n\n"
                    "以下の構造でレスポンス最適化プロンプトを出力してください: \n\n"
                    "目標分析: \n"
                    "[ここに目標の分析結果を記入]\n\n"
                    "レスポンス仕様:\n"
                    "[ここに策定されたレスポンス仕様を記入]\n\n"
                    "AIエージェントへの指示:\n"
                    "[ここにAIエージェントへの具体的な指示を記入]\n\n"
                    "レスポンス例:\n"
                    "[ここにレスポンス例を記入]\n\n"
                    "評価基準:\n"
                    "[ここに評価基準を記入]\n\n"
                    "では、以下の目標にタウするレスポンス最適化プロンプトを作成してください:\n"
                    "{query}",
                )
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query":query})
        

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

    argtask="サイバーセキュリティの市場動向の調査"

    print("-----Passive Goal Creator-----")
    goal_creator = PassiveGoalCreator(llm=llm)
    goal: Goal = goal_creator.run(query=argtask)

    #print(f"{result.text}")

    print("-----Prompt Optimizer-----")    
    prompt_optimizer = PromptOptimizer(llm=llm)
    optimized_goal: OptimizedGoal = prompt_optimizer.run(query=goal.text)

    #print(f"{optimized_goal.text}")

    print("-----Response Optimizer-----")    
    response_optimizer = ResponseOptimizer(llm=llm)
    optimized_response: OptimizedGoal = response_optimizer.run(query=optimized_goal.text)

    print(f"{optimized_response}")
    
    print("----------end------------------")
    
################################################
#
#
if __name__ == "__main__":
    main()
    

