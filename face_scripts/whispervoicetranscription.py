import whisper

model = whisper.load_model("base.en")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("/home/user/VScode_PS1/random_data/Mike Shinoda - Bleed It Out (Already Over Sessions).mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
#_, probs = model.detect_language(mel)
#print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)