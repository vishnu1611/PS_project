import librosa
import whisper
import datetime
import subprocess
import torch
from pyannote.core import Segment
import wave
import contextlib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import parselmouth
import os
from groq import Groq

# Function to format time
def time(secs):
    return datetime.timedelta(seconds=round(secs))

# Function to convert audio to wav format if needed
def convert_to_wav(path):
    if path[-3:] != 'wav':
        output_path = 'audio.wav'
        subprocess.call(['ffmpeg', '-i', path, output_path, '-y'])
        return output_path
    return path

# Function to extract features from audio
def extract_features(waveform, sr):
    sound = parselmouth.Sound(waveform, sampling_frequency=sr)
    formant = sound.to_formant_burg()

    # Get formant frequencies for the middle of the audio
    formants = [formant.get_value_at_time(i, sound.get_total_duration()/2) for i in range(1, 5)]
    formants = np.array(formants)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13).mean(axis=1)

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr).mean(axis=1)

    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=waveform, sr=sr).mean(axis=1)

    # Pitch
    pitch = parselmouth.praat.call(sound, "To Pitch", 0.0, 75, 600)
    pitch_values = pitch.selected_array['frequency'].flatten()
    pitch_mean = np.mean(pitch_values) if pitch_values.size > 0 else 0

    # Harmonic-to-Noise Ratio
    hnr = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 600, 1.0)
    hnr_values = hnr.values.flatten()
    hnr_mean = np.mean(hnr_values) if hnr_values.size > 0 else 0

    # Jitter
    point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
    jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0.0, 0.02, 0.001, 0.03, 1.3)
    jitter_value = jitter if jitter is not None else 0

    # Shimmer
    shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0.0, 0.02, 0.001, 0.03, 1.6, 1.0)
    shimmer_value = shimmer if shimmer is not None else 0

    features = np.hstack((
        formants,
        mfccs,
        spectral_contrast,
        chroma_stft,
        pitch_mean,
        hnr_mean,
        jitter_value,
        shimmer_value
    ))

    return features

# Function to load Whisper model
def load_model(language='English', model_size='large'):
    model_name = model_size
    if language == 'English' and model_size != 'large':
        model_name += '.en'
    model = whisper.load_model(model_size)
   
    return model

# Function to transcribe audio
def transcribe_audio(model, path):
    result = model.transcribe(path)
    return result["segments"]

# Function to get audio duration
def get_audio_duration(path):
    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration

# Function to extract features for audio segments
def extract_segment_features(segments, path):
    segment_features = []
    for segment in segments:
        start_time = segment["start"]
        end_time = segment["end"]
        segment_waveform, sr = librosa.load(path, sr=16000, offset=start_time, duration=end_time - start_time)
        segment_features.append(extract_features(segment_waveform, sr))
    return segment_features

# Function to standardize and apply PCA to features
# def standardize_and_reduce(segment_features, sample_features):
#     combined_features = np.vstack((segment_features, sample_features))
#     scaler = StandardScaler()
#     combined_features = scaler.fit_transform(combined_features)

#     segment_features = combined_features[:len(segment_features)]
#     sample_features = combined_features[len(segment_features):]

#     imputer = SimpleImputer(strategy='mean')
#     segment_features_imputed = imputer.fit_transform(segment_features)
#     sample_features_imputed = imputer.fit_transform(sample_features)

#     #Apply PCA to reduce dimensions
#     pca = PCA(n_components=2)
#     reduced_features = pca.fit_transform(np.vstack((segment_features_imputed, sample_features_imputed)))
#     reduced_segment_features = reduced_features[:len(segment_features)]
#     reduced_sample_features = reduced_features[len(segment_features):]

def standardize_and_reduce(segment_features, sample_features):
    # Print shapes for debugging
    print("Shape of segment_features:", np.array(segment_features).shape)
    print("Shape of sample_features:", np.array(list(sample_features.values())).shape)

    # Convert sample_features dictionary to a numpy array
    sample_features = np.array(list(sample_features.values()))

    combined_features = np.vstack((segment_features, sample_features))
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)

    segment_features = combined_features[:len(segment_features)]
    sample_features = combined_features[len(segment_features):]

    imputer = SimpleImputer(strategy='mean')
    segment_features_imputed = imputer.fit_transform(segment_features)
    sample_features_imputed = imputer.fit_transform(sample_features)

    # Apply PCA to reduce dimensions
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(np.vstack((segment_features_imputed, sample_features_imputed)))
    reduced_segment_features = reduced_features[:len(segment_features)]
    reduced_sample_features = reduced_features[len(segment_features):]

    return reduced_segment_features, reduced_sample_features




# # Split back the standardized features
# segment_features = combined_features[:len(segment_features)]
# sample_features = combined_features[len(segment_features):]

# # Impute NaN values with the mean of the column
# imputer = SimpleImputer(strategy='mean')
# segment_features_imputed = imputer.fit_transform(segment_features)
# sample_features_imputed = imputer.fit_transform(sample_features)

# Apply PCA to reduce dimensions
# pca = PCA(n_components=2)
# reduced_features = pca.fit_transform(np.vstack((segment_features_imputed, sample_features_imputed)))
# reduced_segment_features = reduced_features[:len(segment_features)]
# reduced_sample_features = reduced_features[len(segment_features):]

    return reduced_segment_features, reduced_sample_features

# Function to assign speaker labels based on minimum distance in PCA space
def assign_speaker_labels(segments, reduced_segment_features, reduced_sample_features, sample_audios):
    speaker_mapping = {label: [] for label in sample_audios.keys()}
    for i, segment_feature in enumerate(reduced_segment_features):
        distances = {
            sample_label: euclidean_distances(
                segment_feature.reshape(1, -1),
                reduced_sample_features[j].reshape(1, -1)
            )[0][0]
            for j, sample_label in enumerate(sample_audios.keys())
        }
        assigned_label = min(distances, key=distances.get)
        speaker_mapping[assigned_label].append(i)

    for i, segment in enumerate(segments):
        current_speaker_label = None
        for speaker_label, indices in speaker_mapping.items():
            if i in indices:
                current_speaker_label = speaker_label
                break
        segment["speaker"] = current_speaker_label if current_speaker_label else "Unknown"

    return segments

# Function to save transcript
def save_transcript(segments, path="transcript.txt"):
    with open(path, "w") as f:
        for i, segment in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
            f.write(segment["text"] + ' ')

# Function to extract features from sample audios
def extract_sample_features(sample_audios):
    sample_features = {}
    for label, sample_path in sample_audios.items():
        waveform, sr = librosa.load(sample_path, sr=16000)
        sample_features[label] = extract_features(waveform, sr)
    return sample_features


# Function to summarize transcript
def summarize_transcript(client, transcript_content):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "summarize" + transcript_content,
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content

# Function to list action points in transcript
def list_action_points(client, transcript_content):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "list all the action points in" + transcript_content,
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content
