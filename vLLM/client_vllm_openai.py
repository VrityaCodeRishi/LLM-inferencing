import os

from openai import OpenAI


def main() -> None:
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

    client = OpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Say hello world in one short sentence."},
        ],
        temperature=0,
    )

    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()

