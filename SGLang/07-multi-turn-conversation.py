import sglang as sgl

backend = sgl.RuntimeEndpoint("http://localhost:30000")
sgl.set_default_backend(backend)

@sgl.function
def multi_turn(s, topic):
    s += sgl.system("You are an expert teacher.")
    
    # Turn 1: Introduction
    s += sgl.user(f"Explain {topic} to me like I'm a beginner.")
    s += sgl.assistant(sgl.gen("intro", max_tokens=150))
    
    # Turn 2: Follow-up
    s += sgl.user("Can you give me a practical example?")
    s += sgl.assistant(sgl.gen("example", max_tokens=150))
    
    # Turn 3: Summary
    s += sgl.user("Summarize the key points in 3 bullet points.")
    s += sgl.assistant(sgl.gen("summary", max_tokens=100))

state = multi_turn.run(topic="prefix caching in LLM inference")

print("=== Introduction ===")
print(state["intro"])
print("\n=== Example ===")
print(state["example"])
print("\n=== Summary ===")
print(state["summary"])