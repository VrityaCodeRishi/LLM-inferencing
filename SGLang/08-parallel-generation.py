import sglang as sgl

backend = sgl.RuntimeEndpoint("http://localhost:30000")
sgl.set_default_backend(backend)

@sgl.function
def generate_idea(s, topic):
    s += sgl.system("You are a creative brainstorming assistant.")
    s += sgl.user(f"Give me one unique business idea related to: {topic}")
    s += sgl.assistant(sgl.gen("idea", max_tokens=100, temperature=0.9))

# Method 1: Use run_batch for parallel generation (recommended)
print("=== Generating 3 ideas in parallel ===\n")

states = generate_idea.run_batch([
    {"topic": "sustainable technology"},
    {"topic": "sustainable technology"},
    {"topic": "sustainable technology"},
])

for i, state in enumerate(states):
    print(f"Idea {i+1}: {state['idea']}\n")
    print("-" * 50)