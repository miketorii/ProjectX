import os
from dotenv import load_dotenv
import json
import uuid
from typing import Optional

import faiss
import numpy as np

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from retry import retry

from settings import Settings

##########################################################
#
#
class ReflectionJudgement(BaseModel):
    needs_retry: bool = Field(description="タスクの実行結果は適切だったと思いますか？あなたの判断を真偽値で示してください。")
    confidence: float = Field(description="あなたの判断に対するあなたの自信の度合いを0から1までの少数で示してください。")
    reasons: list[str] = Field(description="タスクの実行結果の適切性とそれに対する自信度について、判断に至った理由を簡潔に列挙してください。")
    
##########################################################
#
#
class Reflection(BaseModel):
    id: str = Field(description="リフレクション内容に一意性を与えるためのID")
    task: str = Field(description="ユーザから与えられたタスクの内容")
    reflection: str = Field(description="このタスクに取り組んだ際の阿多田の思考プロセスを振り返ってください。何か改善できる点はありましたか？次に同様のタスクに取り組む際に、より結果を出すための教訓を2〜3文程度で簡潔に述べてください。")
    judgement: ReflectionJudgement = Field(description="リトライが必要かどうかの判定")

##########################################################
#
#
class ReflectionManager:
    def __init__(self, file_path: str):
        self.file_path = file_path
        conf = Settings()
        conf.readenv()
        self.embeddings = AzureOpenAIEmbeddings(
            model="my-text-embedding-3-large",
            azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDED_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_EMBEDDED_API_KEY"],
            # openai_api_version=AZURE_OPENAI_EMBEDDING_API_VERSION    
            # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
        )
        self.reflections: dict[str, Reflection] = {}
        self.embeddings_dict: dict[str, list[float]] = {}
        self.index = None
        self.load_reflections()
#        print(self.embeddings_dict)
#        print("-----------end init ReflectionManager------")

    def load_reflections(self):
        print("----load_reflections----")
        if os.path.exists(self.file_path):
            with open(self.file_path,"r") as file:
                data = json.load(file)
                for item in data:
                    reflection = Reflection(**item["reflection"])
                    self.reflections[reflection.id] = reflection
                    self.embeddings_dict[reflection.id] = item["embedding"]

            if self.reflections:
                embeddings = list(self.embeddings_dict.values())
                self.index = faiss.IndexFlatL2(len(embeddings[0]))
                self.index.add(np.array(embeddings).astype("float32"))

    def save_reflection(self, reflection: Reflection) -> str:
        reflection.id = str(uuid.uuid4())
        reflection_id = reflection.id
        self.reflections[reflection_id] = reflection
        embedding = self.embeddings.embed_query(reflection.reflection)
        self.embeddings_dict[reflection_id] = embedding

        if self.index is None:
            self.index = faiss.IndexFlatL2(len(embedding))
        self.index.add(np.array([embedding]).astype("float32") )

        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump(
                [
                    {"reflection": reflection.model_dump(), "embedding": embedding}
                    for reflection, embedding in zip(
                            self.reflections.values(), self.embeddings_dict.values()
                    )
                ],
                file,
                ensure_ascii=False,
                indent=4,
            )

        return reflection_id

    def get_reflection(self, reflection_id: str) -> Optional[Reflection]:
        return self.reflections.get(reflection_id)


    def get_relevant_reflections(self, query: str, k: int =3) -> list[Reflection]:
        if not self.reflections or self.index is None:
            return []

        query_embedding = self.embeddings.embed_query(query)
        try:
            D, I = self.index.search(
                np.array([query_embedding]).astype("float32"),
                min(k, len(self.reflections)),
            )
            reflection_ids = list(self.reflections.keys())
            
            return [
                self.reflections[reflection_ids[i]]
                for i in I[0]
                if i < len(reflection_ids)
            ]
        except Exception as e:
            print(f"Error during reflection search: {e}")
            return []
        
##########################################################
#
#
class TaskReflector:
    def __init__(self, llm: BaseChatModel, reflection_manager: ReflectionManager):
        self.llm = llm.with_structured_output(Reflection)
        self.reflection_manager = reflection_manager
        print("--------End init in TaskReflector-------")
        
    def run(self, task: str, result: str) -> Reflection:
        prompt = ChatPromptTemplate.from_template(
            "与えられたタスクの内容:\n{task}\n\n"
            "タスクを実行した結果:\n{result}\n\n"
            "あなたは高度な推論能力を持つAIエージェントです。上記のタスクを実行した結果を分析し、このタスクに対するあなたの取り組みが適切だったかどうかを内省してください。\n"
            "以下の項目に沿って、リフレクションの内容を出力してください。\n\n"
            "リフレクション:\n"
            "このタスクに取り組んだ際のあなたの思考プロセスや方法を振り返ってください。何か改善できる点はありましたか？\n"
            "次に同様のタスクに取り組む際に、より良い結果を出すための教訓を2〜3文程度で簡潔に述べてください。\n\n"
            "判定:\n"
            " - 結果の適切性: タスクの実行結果は適切だったと思いますか？あなたの判断を真偽値で示してください。\n"
            " - 判定の自由度: 上記判断に対するあなたの自信の度合いを0から1までの小数で示してください。\n"
            " - 判定の理由: タスクの実行結果の適切性とそれに対する自信度について、判断に至った理由を簡潔に列挙してください。 \n\n"
            "出力は必ず日本語で行ってください。\n\n"
            "Tips: Make sure to answer in the correct format."
        )

        chain = prompt | self.llm

        @retry(tries=5)
        def invoke_chain() -> Reflection:
            return chain.invoke({"task": task, "result": result})

        reflection = invoke_chain()
        reflection_id = self.reflection_manager.save_reflection(reflection)
        reflection.id = reflection_id
        
        return reflection
    
if __name__=="__main__":
    print("-------------------Start main-------------------------")        
    refmgr = ReflectionManager("tmp/self_reflection_db.json")
    ref_judge = ReflectionJudgement(
        needs_retry=True,
        confidence=0.9,
        reasons=["str1","str2"]
    )
    print(ref_judge)
    ref = Reflection(
        id="100",
        task="リコーがPFUを買収した理由を100文字以内でまとめてください。",
        reflection="this is reflection",
        judgement=ref_judge        
    )
    print("--------------------------------------------")    
    print(ref)
    ref_id = refmgr.save_reflection(ref)
    print("--------------------------------------------")    
    print(f"Reference ID: {ref_id}")
    refgot = refmgr.get_reflection(ref_id)
    print("--------------------------------------------")    
    print(refgot)

    listRef = refmgr.get_relevant_reflections("str3")
    print(listRef)

    llm = AzureChatOpenAI(
        azure_deployment="my-gpt-4o-1",
        api_version="2024-08-01-preview",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    mytask="リコーがPFUを買収した理由を100文字以内でまとめてください。"
    prev_result = "リコーはデジタルサービスの売上拡大のためPFUのスキャンソリューションを獲得したかった。"
    taskref = TaskReflector(llm, refmgr)
    resultref = taskref.run(mytask, prev_result)
    print("--------------------------------------------")
    print(resultref)
    print("-------------------END main-------------------------")    
