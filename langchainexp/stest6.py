import os
import re
#from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate

from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

######################################
#
#    
def extract_last_action_and_input(text):
#    action_pattern = re.compile(r"(?i)action\s*:\s*([^\n]+)", re.MULTILINE)
#    action_input_pattern = re.compile(r"(?i)action\s*_*input\s*:\s*([^\n]+)", re.MULTILINE)

    action_pattern = re.compile(r"(?i)action\s*:\s*([^\n]+)", re.MULTILINE)
    action_input_pattern = re.compile(
        r"(?i)action\s*_*input\s*:\s*([^\n]+)", re.MULTILINE
    )
    
    actions = action_pattern.findall(text)
    action_inputs = action_input_pattern.findall(text)    

    last_action = actions[-1] if actions else None
    last_action_input = action_inputs[-1] if action_inputs else None

    print(last_action+"345")
    print("Last Action: ", last_action)
    print("Last Action Input: ", last_action_input)

    return {"action": last_action, "action_input": last_action_input}

######################################
#
#
def extract_final_answer(text):
    final_answer_pattern = re.compile(r"(?i)I've found the answer:\s*([^\n]+)", re.MULTILINE)
    final_answers = final_answer_pattern.findall(final_answer_text)
    print("Final Answers:", final_answers)
    if final_answers:
        return final_answers[0]
    else:
        return None

######################################
#
#
def search_on_google(query: str):
    return f"Jason Derulo doesn't have a wife or parter."

######################################
#
#    
print("----------start----------------")

text = """
Action: search_on_google
Action_Input: Tom Hanks current wife

action: search_on_wikipedia
action_input: How old is Rita Wilson in 2023

action: search_on_google
action input: some other query
"""

ret = extract_last_action_and_input(text)
print(ret)

final_answer_text = "I've found the answer: final_answer"
ret = extract_final_answer(final_answer_text)
print(ret)

#final_answer_pattern = re.compile(r"(?i)I've found the answer:\s*([^\n]+)", re.MULTILINE)
#final_answers = final_answer_pattern.findall(final_answer_text)
#print("Final Answers:", final_answers)

print("--------------------------")

#action_pattern = re.compile(r"(?i)action\s*:\s*([^\n]+)", re.MULTILINE)
#action_input_pattern = re.compile(r"(?i)action\s*_*input\s*:\s*([^\n]+)", re.MULTILINE)

#actions = action_pattern.findall(text)
#action_inputs = action_input_pattern.findall(text)

#print(actions)
#print(action_inputs)

#last_action = actions[-1] if actions else None
#last_action_input = action_inputs[-1] if action_inputs else None

#print("Last Action: ", last_action)
#print("Last Action Input: ", last_action_input)


model = AzureChatOpenAI(
    azure_deployment="MyModel",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

#client = AzureOpenAI(
#    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
#    api_key=os.getenv("AZURE_OPENAI_KEY"),
#    api_version="2024-08-01-preview"
#)

tools = {}

#tools["search_on_google"] = {
#    "function": search_on_google,
#    "description": "Searches on google for a query"
#}

tools["search_on_google"] = {
    "function": search_on_google,
    "description": "Searches on google for a query",
}

base_prompt = """
You will attempt to solve the problem of finding the answer to a question.
Use chain of thought reasoning to solve through the problem, using the following pattern:

1. Observe the original question:
original_question: original_problem_text
2. Create an observation with the following pattern:
observation: observation_text
3. Create a thought based on the observation with the following pattern:
thought: thought_text
4. Use tools to act on the thought with the following pattern:
action: tool_name
action_input: tool_input

Do not guess or assume the tool results. Instead, provide a structured output that includes the action and action_input.

You have access to the following tools: {tools}.

original_problem: {question}
"""

model_output = model.invoke(
    SystemMessagePromptTemplate.from_template(template=base_prompt).format_messages(
        tools=tools, question="Is Jason Derulo with a partner?"
    )
)
print(model_output)

tool_name = extract_last_action_and_input(model_output.content)["action"]
tool_input = extract_last_action_and_input(model_output.content)["action_input"]
print(tool_name+"123")
print(tool_input)
tool_result = tools["search_on_google"]["function"](tool_input)
print(tool_result)

#tool_result = tools[tool_name]["function"](tool_input)
#tool_result = ""
tool_result = tools[tool_name]["function"](tool_input)

print(
f"""
----------
Below is
name: {tool_name}
input: {tool_input}
result: {tool_result}
----------
"""
)

print("-------------------------------")
print("-------------------------------")
print("----------end------------------")
