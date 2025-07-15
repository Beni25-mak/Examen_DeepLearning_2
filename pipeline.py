import librosa
import torch
import warnings
warnings.filterwarnings("ignore")
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline

print("Chargement du modèle Wav2Vec2 (speech-to-text)...")
asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

print("Chargement du modèle d'analyse de sentiment...")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def transcribe_audio(audio_path, sr=16000):
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        input_values = asr_processor(audio, sampling_rate=sr, return_tensors="pt").input_values
        with torch.no_grad():
            logits = asr_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = asr_processor.decode(predicted_ids[0])
        return transcription.lower()
    except Exception as e:
        print(f"Erreur de transcription : {e}")
        return ""

def predict_sentiment(text):
    if not text.strip():
        return "neutre", 0.0
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

def process_audio_pipeline(audio_path):
    transcription = transcribe_audio(audio_path)
    sentiment, confidence = predict_sentiment(transcription)
    return transcription, sentiment, confidence