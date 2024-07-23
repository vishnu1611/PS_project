# Upload audio file

import whisper
import datetime
import subprocess
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
import numpy as np



# Parameters
num_speakers = 2  # Set the number of speakers
language = 'English'  # Set the language
model_size = 'base'  # Set the Whisper model size

# Adjust model name based on language and size
model_name = model_size
if language == 'English' and model_size != 'large':
    model_name += '.en'



# Load Whisper model
model = whisper.load_model(model_size)

# Function to convert segment to embedding
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))

# Convert audio to wav format if needed
if path[-3:] != 'wav':
    subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
    path = 'audio.wav'

# Transcribe audio
result = model.transcribe(path)
segments = result["segments"]

# Get audio duration
with contextlib.closing(wave.open(path, 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

# Initialize audio object
audio = Audio()

# Function to extract segment embedding
def segment_embedding(segment):
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    return embedding_model(waveform[None])

# Create embeddings for each segment
embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
    embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)

# Perform clustering to identify speakers
clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_
for i in range(len(segments)):
    segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

# Function to format time
def time(secs):
    return datetime.timedelta(seconds=round(secs))

# Write transcript with speaker labels to file
with open("transcript.txt", "w") as f:
    for i, segment in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
        f.write(segment["text"][1:] + ' ')

print("Transcript with speaker labels saved to transcript.txt")
