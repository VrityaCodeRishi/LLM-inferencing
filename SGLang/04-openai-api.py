from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="default",  # matches --served-model-name in docker-compose
    messages=[
        {"role": "user", "content": "What is prefix caching and why is it useful?"}
    ],
    max_tokens=200,
    temperature=0.7
)

print(response.choices[0].message.content)

## Streaming with OpenAI
# from openai import OpenAI

# client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

# stream = client.chat.completions.create(
#     model="default",
#     messages=[{"role": "user", "content": "Count from 1 to 10 slowly"}],
#     max_tokens=100,
#     stream=True
# )

# for chunk in stream:
#     if chunk.choices[0].delta.content:
#         print(chunk.choices[0].delta.content, end="", flush=True)