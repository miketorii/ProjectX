import os
from dotenv import load_dotenv
import json
import uuid

import faiss
import numpy as np

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from langchain_openai import AzureOpenAIEmbeddings

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
                    {"reflection": reflection.dict(), "embedding": embedding}
                    for reflection, embedding in zip(
                            self.reflections.values(), self.embeddings_dict.values()
                    )
                ],
                file,
                ensure_ascii=False,
                indent=4,
            )

        return reflection_id
    
##########################################################
#
#
class TaskReflector:
    def __init__(self, llm: BaseChatModel, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        

if __name__=="__main__":
    refmgr = ReflectionManager("tmp/self_reflection_db.json")
    ref_judge = ReflectionJudgement(
        needs_retry=True,
        confidence=0.9,
        reasons=["str1","str2"]
    )
    print(ref_judge)
    ref = Reflection(
        id="100",
        task="task a",
        reflection="this is reflection",
        judgement=ref_judge        
    )
    print(ref)
    ret = refmgr.save_reflection(ref)
    print(ret)
