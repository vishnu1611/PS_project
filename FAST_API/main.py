from fastapi import FastAPI, File, UploadFile
import whisper
import wave
import contextlib
import torch
import torchaudio
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier

app = FastAPI()

# Load models
num_speakers = 2 #@param {type:"integer"}

language = 'English' #@param ['any', 'English']

model_size = 'large' #@param ['tiny', 'base', 'small', 'medium', 'large']


model_name = model_size
if language == 'English' and model_size != 'large':
  model_name += '.en'


def segment_embedding(segment, classifier, signal, fs):
    # Function to compute the embedding for a given segment
    start = int(segment['start'] * fs)
    end = int(segment['end'] * fs)
    segment_signal = signal[:, start:end]
    embedding = classifier.encode_batch(segment_signal)
    return embedding.squeeze().numpy()

@app.post("/diarize/")
async def diarize(file: UploadFile = File(...), num_speakers: int = 2):
    # Save uploaded file
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    # Transcribe the audio file using Whisper
    result = model.transcribe(file_location)
    segments = result["segments"]

    # Load the audio file for embedding extraction
    signal, fs = torchaudio.load(file_location)

    # Calculate embeddings
    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment, classifier, signal, fs)

    # Handle NaN values
    embeddings = np.nan_to_num(embeddings)

    # Perform clustering
    clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings)
    labels = clustering.labels_

    # Assign speaker labels to segments
    for i in range(len(segments)):
        segments[i]["speaker"] = f'SPEAKER {labels[i] + 1}'

    # Return the segments with speaker labels
    return {"segments": segments}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)