import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.prompts import ChatPromptTemplate

print("----------start----------------")

class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")

output_parser = PydanticOutputParser(pydantic_object=Recipe)

format_instructions = output_parser.get_format_instructions()
print(format_instructions)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "ユーザが入力した料理のレシピを考えてください\n\n"
            "{format_instructions}",
        ),  
        ( "human", "{dish}"),
    ]
)

prompt_with_format_instructions = prompt.partial(
    format_instructions = format_instructions
)

prompt_value = prompt_with_format_instructions.invoke({"dish":"カレー"})
print("========system=======")
print(prompt_value.messages[0].content)
print("========user=======")
print(prompt_value.messages[1].content)

print("====================")
llm = AzureChatOpenAI(
    azure_deployment="my-gpt-4o-1",
    api_version="2024-08-01-preview",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

ai_msg = llm.invoke(prompt_value)

#print(ai_msg)
print(ai_msg.content)

print("====================")
recipe = output_parser.invoke(ai_msg)
print(type(recipe))
print(recipe)


print("----------end------------------")

