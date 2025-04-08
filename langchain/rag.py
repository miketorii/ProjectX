from langchain_community.document_loaders import GitLoader

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

import os
from settings import Settings

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.documents import Document

from pydantic import BaseModel, Field

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
def process1():
    print("----------start----------------")

    loader = GitLoader(
        clone_url="https://github.com/miketorii/ProjectX",
        repo_path="./tmpgitdata",
        branch="master",
        file_filter=file_filter,
    )

    documents = loader.load()
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

    db = Chroma.from_documents(documents, embeddings)

    prompt = ChatPromptTemplate.from_template(
        "以下の文脈だけを踏まえて質問に回答してください。\n\n"
        "文脈:{context}\n\n"
        "質問: {question}\n"
    )
    
    llm = AzureChatOpenAI(
        azure_deployment="my-gpt-4o-1",
        api_version="2024-08-01-preview",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    retriever = db.as_retriever()

    chain = {"question": RunnablePassthrough(), "context": retriever } | prompt | llm | StrOutputParser()

    querystr = "LangChainの概要を教えて"
    result = chain.invoke(querystr)

    print("----------Final result-----------------------")
    print(result)
    
    print("----------end------------------")

################################################
#
#
class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")
    
################################################
#
#
def process2():
    print("----------start process2----------------")

    loader = GitLoader(
        clone_url="https://github.com/miketorii/ProjectX",
        repo_path="./tmpgitdata",
        branch="master",
        file_filter=file_filter,
    )

    documents = loader.load()
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

    db = Chroma.from_documents(documents, embeddings)

    query_generation_prompt = ChatPromptTemplate.from_template(
        "質問に対してベクターデータベースから関連文書を検索するために、\n"
        "３つの異なる検索クエリを生成してください。\n"
        "距離ベースの類似性検索の限界を克服するために、\n"
        "ユーザーの質問に対して複数の視点を提供することが目標です。\n\n"        
        "質問: {question}\n"
    )
    
    llm = AzureChatOpenAI(
        azure_deployment="my-gpt-4o-1",
        api_version="2024-08-01-preview",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    retriever = db.as_retriever()

    query_generation_chain = (
        query_generation_prompt
        | llm.with_structured_output(QueryGenerationOutput)
        | (lambda x: x.queries)
    )

    prompt = ChatPromptTemplate.from_template(
        "以下の文脈だけを踏まえて質問に回答してください。\n\n"
        "文脈:{context}\n\n"
        "質問: {question}\n"
    )
    
    multi_query_rag_chain = {
        "question": RunnablePassthrough(),
        "context": query_generation_chain | retriever.map(),
    } | prompt | llm | StrOutputParser()

    querystr = "LangChainの概要を教えて"
    result = multi_query_rag_chain.invoke(querystr)

    print("----------Final result-----------------------")
    print(result)
    
    print("----------end process2------------------")

    print("-------process 3---------")

    rag_fusion_chain = {
        "question": RunnablePassthrough(),
        "context": query_generation_chain | retriever.map() | reciprocal_rank_fusion
    } | prompt | llm | StrOutputParser()

    result3 = rag_fusion_chain.invoke(querystr)
    print(result3)

    print("----------end process3------------------")
    
################################################
#
#
def reciprocal_rank_fusion(
        retriever_outputs: list[list[Document]],
        k: int = 60
) -> list[str]:
    content_score_mapping = {}

    for docs in retriever_outputs:
        for rank, doc in enumerate(docs):
            content = doc.page_content

            if content not in content_score_mapping:
                content_score_mapping[content] = 0

            content_score_mapping[content] += 1 / (rank+k)

    ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)
    
    return [content for content, _ in ranked]

################################################
#
#    
def process3():
    print("-------process 3 empty---------")
    
    
################################################
#
#    
def process4():
    print("-------process 4---------")
    
################################################
#
#
def main(exe_num: int):
    if exe_num == 1:
        process1()
    elif exe_num == 2:
        process2()
    elif exe_num == 3:
        process3()
    else:
        process4()    
    

################################################
#
#
if __name__ == "__main__":
    main(2)
    

