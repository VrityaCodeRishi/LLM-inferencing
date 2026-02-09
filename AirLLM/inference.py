import torch
from airllm import AutoModel
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-72B"
MAX_NEW_TOKENS = 100
USE_4BIT = True
print(USE_4BIT)

print("Loading model via AirLLM (layer-by-layer mode)...")

if USE_4BIT:
    print("using 4bit")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        compression="4bit",
    )
else:
    model = AutoModel.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def generate_response(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    print(f"\nPrompt: {prompt}")
    print("Generating (this takes a few minutes for 72B)...\n")

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=256,
    )["input_ids"].to("cuda")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


if __name__ == "__main__":
    prompts = [
        "What are the key differences between transformers and RNNs?",
    ]

    for prompt in prompts:
        response = generate_response(prompt)
        print("=" * 60)
        print(response)
        print("=" * 60)
        print()