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

from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableParallel

################################################
#
#
def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

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

print("-------Start Hybrid Search---------")

################################################
################################################

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
    
################################################
################################################

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

prompt = ChatPromptTemplate.from_template(
        "以下の文脈だけを踏まえて質問に回答してください。\n\n"
        "文脈:{context}\n\n"
        "質問: {question}\n"
)

chroma_retriever = retriever.with_config(
    {"run_name": "chroma_retriever"}
)

bm25_retriever = BM25Retriever.from_documents(documents).with_config(
    {"run_name": "bm25_retriever"}
)

hybrid_retriever = (
    RunnableParallel({
        "chroma_documents": chroma_retriever,
        "bm25_documents": bm25_retriever
    })
    | (lambda x: [x["chroma_documents"],x["bm25_documents"]] )
    | reciprocal_rank_fusion
)

hybrid_rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": hybrid_retriever
        }
        | prompt | llm | StrOutputParser()
)

querystr = "LangChainの概要を教えて"

result = hybrid_rag_chain.invoke(querystr)
print(result)

    
print("-------END Hybrid Search---------")    
    
    

