import requests

url = "http://localhost:30000/v1/chat/completions"

response = requests.post(url, json={
    "model": "default",  # matches --served-model-name in docker-compose
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what RadixAttention does in one paragraph."}
    ],
    "max_tokens": 150,
    "temperature": 0.7
})

result = response.json()
print(result["choices"][0]["message"]["content"])