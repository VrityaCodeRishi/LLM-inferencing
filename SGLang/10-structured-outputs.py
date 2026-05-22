import sglang as sgl
import json

backend = sgl.RuntimeEndpoint("http://localhost:30000")
sgl.set_default_backend(backend)

# Define JSON schema
person_schema = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "occupation": {"type": "string"},
        "skills": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["name", "age", "occupation", "skills"]
})

@sgl.function
def extract_person(s, text):
    s += sgl.user(f"Extract person information from this text as JSON: {text}")
    s += sgl.assistant(sgl.gen("person", json_schema=person_schema, max_tokens=200))

state = extract_person.run(
    text="John Smith is a 35 year old software engineer who specializes in Python, Go, and cloud architecture."
)

person = json.loads(state["person"])
print(json.dumps(person, indent=2))