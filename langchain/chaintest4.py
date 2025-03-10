from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

print("----------start----------------")

class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザが入力した料理のレシピを考えてください"),  
        ( "human", "{dish}"),
    ]
)

model = AzureChatOpenAI(
    azure_deployment="my-gpt-4o-1",
    api_version="2024-08-01-preview",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

chain = prompt | model.with_structured_output(Recipe)

recipe = chain.invoke({"dish":"カレー"})

print(type(recipe))
print(recipe)

print("======================================")
print(recipe.ingredients)
print("======================================")
print(recipe.steps)

print("----------end------------------")

