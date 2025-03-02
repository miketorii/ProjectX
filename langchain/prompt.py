from langchain_openai import AzureChatOpenAI


print("------------start---------------")

#################################################################
#
# Create client for LLM
#
llm = AzureChatOpenAI(
    #azure_deployment="MyModel",  # or your deployment
    #api_version="2024-02-01",  # or your api version
    azure_deployment="my-gpt-4o-1",  # or your deployment    
    api_version="2024-08-01-preview",  # or your api version
    temperature=0.7,
    streaming=True,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

messages = [
    ( "user", "" ),
    ( "human", "プロンプトエンジニアリングとは？" )
]
    
response = llm.invoke(messages)

print(response.content)

print("------------end---------------")

