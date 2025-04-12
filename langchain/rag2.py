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

################################################
#
#
def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

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

    print("------------in routed_retriever-----------")
    print(question)
    print(route)

    if route == Route.langchain_document:
        result_doc = langchain_document_retriever.invoke(question)
        print("-----------called langchain_document_retriever.invoke-------------")        
        return result_doc
    elif route == Route.web:
        result_web = web_retriever.invoke(question)
        print("-----------called web_retriever.invoke-------------")        
        return result_web

    raise ValueError(f"Unknown route: {route}")

################################################
#
#    

print("-------Start process 3---------")


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

print("-------called route_rag_chain-----------")
    
result2 = route_rag_chain.invoke(querystr2)
print(result2)
    
print("-------END process 3---------")    
    
    

