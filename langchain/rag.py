from langchain_community.document_loaders import GitLoader

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

import os
from settings import Settings
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.documents import Document

from pydantic import BaseModel, Field

from langchain_cohere import CohereRerank

from langchain_community.retrievers import TavilySearchAPIRetriever
from enum import Enum

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

    print("----------process4------------------")    
    rerank_rag_chain = (
        {
            "question": RunnablePassthrough(),
            "documents": retriever
        }
        | RunnablePassthrough.assign(context=rerank)
        | prompt | llm | StrOutputParser()
    )

    result4 = rerank_rag_chain.invoke(querystr)
    print(result4)

    print("----------end process4------------------")
    
################################################
#
#
def rerank(inp: dict[str, Any], top_n: int = 3) -> list[Document]:
    question = inp["question"]
    documents = inp["documents"]

    cohere_api_key =os.environ["COHERE_API_KEY"]
    
    cohere_reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)
    
    return cohere_reranker.compress_documents(documents=documents, query=question)

    
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
class Route(str, Enum):
    langchain_document = "langchain_document"
    web = "web"

class RouteOutput(BaseModel):
    route: Route
    
################################################
#
#
def routed_retriever(inp: dict[str, Any]) -> list[Document]:
    question = inp["question"]
    route = inp["route"]

    loader = GitLoader(
        clone_url="https://github.com/miketorii/ProjectX",
        repo_path="./tmpgitdata",
        branch="master",
        file_filter=file_filter,
    )

    documents = loader.load()
    
    conf = Settings()
    conf.readenv()

    embeddings = AzureOpenAIEmbeddings(
        model="my-text-embedding-3-large",
        azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDED_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_EMBEDDED_API_KEY"],
    )
    
    db = Chroma.from_documents(documents, embeddings)
    
    retriever = db.as_retriever()    

    langchain_document_retriever = retriever.with_config(
        {"run_name":"langchain_document_retriever"}
    )

    web_retriever = TavilySearchAPIRetriever(k=3).with_config(
        {"run_name":"web_retriever"}
    )
    
    if route == Route.langchain_document:
        return langchain_document_retriever.invoke(question)
    elif route == Route.web:
        return web_retriever.invoke(question)

    raise ValueError(f"Unknown route: {route}")
    
################################################
#
#    
def process3():
    print("-------process 3---------")

#    langchain_document_retriever = retriever.with_config(
#        {"run_name":"langchain_document_retriever"}
#    )
#
#    web_retriever = TavilySearchAPIRetriever(k=3).with_config(
#        {"run_name":"web_retriever"}
#    )
    
    route_prompt = ChatPromptTemplate.from_template(
        "質問に回答するために適切なRetrieverを選択してください。\n\n"
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
    
    route_chain = (
        route_prompt
        | llm.with_structured_output(RouteOutput)
        | (lambda x: x.route)
    )

    prompt = ChatPromptTemplate.from_template(
        "以下の文脈だけを踏まえて質問に回答してください。\n\n"
        "文脈:{context}\n\n"
        "質問: {question}\n"
    )
    
    route_rag_chain = (
        {
            "question": RunnablePassthrough(),
            "route": route_chain
        }
        | RunnablePassthrough.assign(context=routed_retriever)
        | prompt | llm | StrOutputParser()
    )

    querystr = "LangChainの概要を教えて"
    querystr2 = "東京の今日の天気は？"

    result = route_rag_chain.invoke(querystr)
    print(result)

    print("--------------------------------")
    
    result2 = route_rag_chain.invoke(querystr2)
    print(result2)
    
    print("-------END process 3---------")    
    
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
    main(3)
    

