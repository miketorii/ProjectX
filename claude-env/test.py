import os
import anthropic

print("------start---------")
key = os.environ['ANTHROPIC_API_KEY']

client = anthropic.Anthropic(api_key=key)

# Claudeモデルを呼び出す
response = client.completions.create(
    model="claude-v1",
    max_tokens_to_sample=100,
    prompt="Hello, Anthropic!",
)

# レスポンスを表示
print(response.completion)


'''
print("------message---------")
response = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello, world"}]
)

print(response)

'''


