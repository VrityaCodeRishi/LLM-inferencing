import requests

url = "http://localhost:30000/generate"

response = requests.post(url, json={
    "text": "How Hinata in Haikyuu like passion,motivation and hard workcan help in learning AI Engineering?",
    "sampling_params": {
        "max_new_tokens": 150,
        "temperature": 0.8,
        "top_p": 0.95
    }
})

result = response.json()
print("Generated text:", result["text"])
print("Tokens generated:", result["meta_info"]["completion_tokens"])
