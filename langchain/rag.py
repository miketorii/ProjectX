from langchain_community.document_loaders import GitLoader

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

import os
from settings import Settings

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent



################################################
#
#

################################################
#
#
'''
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
                            "次のタスクを実行し、詳細な回答を提供してください。\n\n"
                            f"タスク: {task} \n\n"
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
'''
################################################
#
#
def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


################################################
#
#
def main():
    print("----------start----------------")

    loader = GitLoader(
        clone_url="https://github.com/miketorii/ProjectX/langchain/testdoc",
        repo_path="./tmpgitdata",
        branch="master",
        file_filter=file_filter,
    )

    documents = loader.load()
    print(documents)
    print(len(documents))

    conf = Settings()
    conf.readenv()
    embeddings = AzureOpenAIEmbeddings(
        model="my-text-embedding-3-large",
        azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDED_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_EMBEDDED_API_KEY"],
        # openai_api_version=AZURE_OPENAI_EMBEDDING_API_VERSION
        # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
    )

    print("------------Chroma in main-------")
    db = Chroma.from_documents(documents, embeddings)
    print("------------END Chroma in main-------")
    
    llm = AzureChatOpenAI(
        azure_deployment="my-gpt-4o-1",
        api_version="2024-08-01-preview",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


    print("----------Final result-----------------------")

    print("----------end------------------")
    
################################################
#
#
if __name__ == "__main__":
    main()
    

