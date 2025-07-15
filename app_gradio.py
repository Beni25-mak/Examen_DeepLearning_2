import gradio as gr
from transcription_vocal import transcribe_audio
from sentiment_analyse import predict_sentiment

def process_audio_gradio(audio_file):
    transcription = transcribe_audio(audio_file)
    if transcription.strip() == "":
        return "Erreur lors de la transcription.", "", ""
    sentiment, confidence = predict_sentiment(transcription)
    return transcription, sentiment + f" ({confidence*100:.1f}%)"
 

# Interface Gradio
iface = gr.Interface(
    fn=process_audio_gradio,
    inputs=gr.Audio(type="filepath", label="Téléverser un fichier audio (.wav)"),
    outputs=[
        gr.Textbox(label="Texte transcrit"),
        gr.Textbox(label="Sentiment détecté")
    ],
    title="Détection automatique de sentiment à partir d'un appel vocal",
    description="Cette application transcrit un fichier audio (voix) en texte avec Wav2Vec2 et analyse le sentiment avec BERT."
)

if __name__ == "__main__":
    iface.launch()