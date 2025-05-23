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

    goal_creator = PassiveGoalCreator(llm=llm)
    result: Goal = goal_creator.run(query="4Pによる新しいプリンターのマーケティング")

    print(f"{result.text}")
    
    print("----------end------------------")
    
################################################
#
#
if __name__ == "__main__":
    main()
    

