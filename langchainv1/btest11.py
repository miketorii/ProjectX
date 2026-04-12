from langchain_core.prompts import PromptTemplate, load_prompt

prompt = PromptTemplate(
    template="Translate this sentence from English to Spanish.\nSentence: {sentence}\nTranslation:",
    input_variables=["sentence"],
)

prompt.save("translation_prompt.json")

loadprompt = load_prompt("translation_prompt.json")

print(loadprompt)
