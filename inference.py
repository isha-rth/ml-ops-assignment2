from transformers import pipeline

MODEL_ID = "IshaIIT/distilbert-news-classifier"

classifier = pipeline(
    "text-classification",
    model=MODEL_ID
)

text = "Apple launches a new AI-powered iPhone."

result = classifier(text)

print(result)
