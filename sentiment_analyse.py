from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']

    if label == 'POSITIVE':
        sentiment = 'satisfait'
    elif label == 'NEGATIVE':
        sentiment = 'mécontent'
    else:
        sentiment = 'neutre'
        
    return sentiment, score