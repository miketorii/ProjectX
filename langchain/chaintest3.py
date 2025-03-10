from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

print("----------start----------------")

class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")

output_parser = PydanticOutputParser(pydantic_object=Recipe)
    
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザが入力した料理のレシピを考えてください\n\n{format_instructions}"),  
        ( "human", "{dish}"),
    ]
)

prompt_with_format_instructions = prompt.partial(
    format_instructions=output_parser.get_format_instructions()
)

model = AzureChatOpenAI(
    azure_deployment="my-gpt-4o-1",
    api_version="2024-08-01-preview",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
).bind( response_format={"type":"json_object"} )

chain = prompt_with_format_instructions | model | output_parser

recipe = chain.invoke({"dish":"カレー"})

print(type(recipe))
print(recipe)

print("======================================")
print(recipe.ingredients)
print("======================================")
print(recipe.steps)

print("----------end------------------")

