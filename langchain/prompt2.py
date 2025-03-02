from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

print("------------start---------------")

#################################################################
#
# Create client for LLM
#
model = AzureChatOpenAI(
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

system_template = "Please answer in {language}:"

user_template = """
Please answer the following question: 
{question}:"""

parser = StrOutputParser()
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", user_template)]
)
chain = prompt_template | model | parser

response = chain.invoke({"language": "japanese", "question": "What is the highest mountain?"})

print(response)

print("------------end---------------")

