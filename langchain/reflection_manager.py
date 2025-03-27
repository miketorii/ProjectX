from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

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

        
##########################################################
#
#
class TaskReflector:
    def __init__(self, llm: BaseChatModel, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        
