from openai import OpenAI
import concurrent.futures

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

prompts = [
    "What is machine learning?",
    "Explain neural networks briefly.",
    "What is deep learning?",
    "Define natural language processing.",
    "What are transformers in AI?"
]

def process_prompt(prompt):
    response = client.chat.completions.create(
        model="default",  # matches --served-model-name in docker-compose
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return prompt, response.choices[0].message.content

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(process_prompt, prompts))

for prompt, answer in results:
    print(f"Q: {prompt}\nA: {answer}\n{'='*50}")
