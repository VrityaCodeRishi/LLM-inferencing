import sglang as sgl

backend = sgl.RuntimeEndpoint("http://localhost:30000")
sgl.set_default_backend(backend)

@sgl.function
def simple_qa(s, question):
    s += sgl.system("You are a helpful AI assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=200))


state = simple_qa.run(question="What makes SGLang fast?")
print(state["answer"])