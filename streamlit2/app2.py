import streamlit as st
from audio_processing2 import (
    load_model, convert_to_wav, transcribe_audio, get_audio_duration,
    extract_segment_features, extract_sample_features, 
    standardize_and_reduce, assign_speaker_labels, save_transcript,
    summarize_transcript, list_action_points
)
import os
from groq import Groq

# Initialize Groq client with API key
GROQ_API_KEY = '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'
client = Groq(api_key=GROQ_API_KEY)

st.title("Audio Transcription and Speaker Labeling")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    path = convert_to_wav(uploaded_file.name)
    duration = get_audio_duration(path)
    st.write(f"Audio Duration: {duration} seconds")

    num_speakers = st.number_input("Enter the number of speakers", min_value=1, step=1)
    sample_audios = {}
    for i in range(num_speakers):
        sample_file = st.file_uploader(f"Upload sample audio for speaker {i + 1}", type=["wav", "mp3", "m4a"], key=f"sample_{i}")
        sample_label = st.text_input(f"Enter label for speaker {i + 1}", key=f"label_{i}")
        if sample_file is not None and sample_label:
            sample_path = convert_to_wav(sample_file.name)
            sample_audios[sample_label] = sample_path

    if st.button("Process Audio"):
        

        model = load_model()
        segments = transcribe_audio(model, path)
        duration = get_audio_duration(path)
        
        sample_features = extract_sample_features(sample_audios)
        segment_features = extract_segment_features(segments, path)
        
        reduced_segment_features, reduced_sample_features = standardize_and_reduce(segment_features, sample_features)
        labeled_segments = assign_speaker_labels(segments, reduced_segment_features, reduced_sample_features, sample_audios)
        
        save_transcript(labeled_segments)
        st.success("Transcription and speaker labeling complete. Transcript saved to transcript.txt.")

        with open("transcript.txt", "r") as f:
            transcript_content = f.read()
            st.text_area("Transcript", transcript_content, height=300)
        
        if st.button("Summarize Transcript"):
            summary = summarize_transcript(client, transcript_content)
            st.text_area("Summary", summary, height=200)

        if st.button("List Action Points"):
            action_points = list_action_points(client, transcript_content)
            st.text_area("Action Points", action_points, height=200)
