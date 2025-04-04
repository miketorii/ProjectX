from langchain_openai import AzureChatOpenAI
from reflection_manager import Reflection,ReflectionManager, TaskReflector
from self_reflection import ReflectiveAgent

################################################
#
#
def main():
    print("----------start----------------")


    llm35 = AzureChatOpenAI(
        azure_deployment="MyModel",  # or your deployment
        api_version="2024-02-01",  # or your api version
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    
    llm4o = AzureChatOpenAI(
        azure_deployment="my-gpt-4o-1",
        api_version="2024-08-01-preview",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    argtask = "ポスト量子暗号の動向について教えてください"
    
    reflection_manager = ReflectionManager(file_path="tmp/cross_reflection_db.json")

    gpt35_task_reflector = TaskReflector(
        llm = llm4o, reflection_manager = reflection_manager
    )

    agent = ReflectiveAgent(
        llm=llm4o,
        reflection_manager=reflection_manager,
        task_reflector=gpt35_task_reflector
    )

    result = agent.run(argtask)
    
    print("----------Final result-----------------------")
    print(result)
    
    print("----------end------------------")
    
################################################
#
#
if __name__ == "__main__":
    main()
