from langchain_core.language_models.chat_models import BaseChatModel


class ReflectionManager:
    def __init__(self, file_path: str):
        self.file_path = file_path

        

class TaskReflector:
    def __init__(self, llm: BaseChatModel, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        
