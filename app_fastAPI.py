from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid
import sys
import logging

# Chemin vers le pipeline et appel des fonction

sys.path.append(os.path.abspath("."))
from transcription_vocal import transcribe_audio
from sentiment_analyse import predict_sentiment


# Configuration basique du logger
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Détection de Sentiment dans un Appel Vocal",
    description="Il s'agit d'un API qui permet de transcrire un fichier audio et prédire le sentiment du client (négatif, neutre, positif).",
    version="1.0"
)

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de détection de sentiment vocal. Utilisez /analyse/ pour poster un fichier .wav"}

@app.post("/analyse/")
async def analyse_audio(file: UploadFile = File(...)):
    logging.info(f"Reçu fichier: {file.filename}")

    # Vérifier l'extension du fichier
    if not file.filename.lower().endswith(".wav"):
        logging.warning(f"Fichier rejeté (extension non supportée) : {file.filename}")
        raise HTTPException(status_code=400, detail="Seuls les fichiers .wav sont acceptés.")

    dossier_temp = "fichier_charge"
    os.makedirs(dossier_temp, exist_ok=True)
    temp_filename = os.path.join(dossier_temp, f"{uuid.uuid4()}.wav")

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Fichier temporaire sauvegardé sous {temp_filename}")

        transcription = transcribe_audio(temp_filename)
        sentiment, confidence = predict_sentiment(transcription)

        logging.info(f"Transcription: {transcription}")
        logging.info(f"Sentiment: {sentiment}, Confiance: {confidence}")

        return {
            "transcription": transcription,
            "sentiment": sentiment,
            "confiance": round(confidence, 2)
        }

    except Exception as e:
        logging.error(f"Erreur lors du traitement : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur: {e}")

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            logging.info(f"Fichier temporaire supprimé : {temp_filename}")