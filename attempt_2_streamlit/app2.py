import streamlit as st

from audio_processing import (
    convert_to_wav, load_model, transcribe_audio,
    get_audio_duration, extract_mfccs, extract_sample_mfccs,
    assign_speaker_labels, save_transcript
)
from pyannote.audio import Audio

st.title("Audio Transcription with Speaker Labeling")

# Upload the audio file
uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a'])
if uploaded_file is not None:
    path = uploaded_file.name
    with open(path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    path = convert_to_wav(path)
    
    # Enter the number of speakers and upload their sample audios
    num_speakers = st.number_input("Enter the number of speakers:", min_value=1, max_value=10, step=1)
    sample_audios = {}
    for i in range(num_speakers):
        sample_file = st.file_uploader(f"Upload sample audio for speaker {i + 1}")
        if sample_file is not None:
            sample_path = sample_file.name
            with open(sample_path, 'wb') as f:
                f.write(sample_file.getbuffer())
            sample_label = st.text_input(f"Enter label for speaker {i + 1}:")
            sample_audios[sample_label] = sample_path

    if st.button("Transcribe"):
        # Load model
        model, embedding_model = load_model()

        # Transcribe audio
        segments = transcribe_audio(model, path)
        duration = get_audio_duration(path)

        # Extract MFCC features
        audio = Audio()
        segment_mfccs = extract_mfccs(audio, path, duration, segments)
        sample_mfccs = extract_sample_mfccs(sample_audios)

        # Assign speaker labels
        segments = assign_speaker_labels(segments, segment_mfccs, sample_mfccs)

        # Save and display the transcript
        save_transcript(segments)
        st.text("Transcript saved to transcript.txt")
