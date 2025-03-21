from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

################################################
#
#
class SinglePathPlanGeneration:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm

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
    

