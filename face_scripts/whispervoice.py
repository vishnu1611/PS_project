import whisper

model = whisper.load_model("base")
result = model.transcribe("/home/user/VScode_PS1/random_data/Mike Shinoda - Bleed It Out (Already Over Sessions).mp3")
print(result["text"])