import sglang as sgl

backend = sgl.RuntimeEndpoint("http://localhost:30000")
sgl.set_default_backend(backend)

@sgl.function
def sentiment_analysis(s, text):
    s += sgl.user(f"Classify the sentiment of this text as positive, negative, or neutral: '{text}'")
    s += sgl.assistant("The sentiment is: " + sgl.gen("sentiment", choices=["positive", "negative", "neutral"]))

state = sentiment_analysis.run(text="I mean it's okay.")
print(f"Sentiment: {state['sentiment']}")