import librosa
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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def time(secs):
    return datetime.timedelta(seconds=round(secs))

def convert_to_wav(path):
    if path[-3:] != 'wav':
        output_path = 'audio.wav'
        subprocess.call(['ffmpeg', '-i', path, output_path, '-y'])
        return output_path
    return path

def segment_mfcc(segment, audio, path, duration):
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    waveform = waveform.numpy().flatten()  # Convert to numpy array
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # Take the mean of the MFCC features
    return mfcc_mean

def calculate_similarity(segment_mfccs, sample_mfccs):
    similarities = {}
    for label, sample_mfcc in sample_mfccs.items():
        similarity = cosine_similarity(
            segment_mfccs.reshape(1, -1),
            sample_mfcc.reshape(1, -1)
        )[0][0]
        similarities[label] = similarity
    assigned_label = max(similarities, key=similarities.get)
    return assigned_label

def load_model(language='English', model_size='small'):
    model_name = model_size
    if language == 'English' and model_size != 'large':
        model_name += '.en'
    model = whisper.load_model(model_size)
    embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
    return model, embedding_model

def transcribe_audio(model, path):
    result = model.transcribe(path)
    return result["segments"]

def get_audio_duration(path):
    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration

def extract_mfccs(audio, path, duration, segments):
    segment_mfccs = np.zeros((len(segments), 13))
    for i, segment in enumerate(segments):
        segment_mfccs[i] = segment_mfcc(segment, audio, path, duration)
    segment_mfccs = np.nan_to_num(segment_mfccs)
    return segment_mfccs

def extract_sample_mfccs(sample_audios):
    sample_mfccs = {}
    for label, sample_path in sample_audios.items():
        y, sr = librosa.load(sample_path, sr=None, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)  # Take the mean of the MFCC features
        sample_mfccs[label] = mfcc_mean
    return sample_mfccs

def assign_speaker_labels(segments, segment_mfccs, sample_mfccs):
    for i, segment_mfcc in enumerate(segment_mfccs):
        assigned_label = calculate_similarity(segment_mfcc, sample_mfccs)
        segments[i]["speaker"] = assigned_label
    return segments

def save_transcript(segments):
    with open("transcript.txt", "w") as f:
        for i, segment in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
            f.write(segment["text"][1:] + ' ')
    print("Transcript with speaker labels saved to transcript.txt")
