import requests
import json

url = "http://localhost:30000/generate"

response = requests.post(url, json={
    "text": "Tell me a short story about a robot:",
    "sampling_params": {"max_new_tokens": 200},
    "stream": True
}, stream=True)

for line in response.iter_lines():
    if line:
        line = line.decode("utf-8")
        if line.startswith("data:"):
            if line == "data: [DONE]":
                break
            data = json.loads(line[5:].strip())
            if "text" in data:
                print(data["text"], end="", flush=True)

print()