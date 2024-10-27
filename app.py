import os
import torch
import whisper

model_path = "whisper_model_small.pt"

# Check if the model file exists
if os.path.exists(model_path):
    # Load the model from the file
    model = torch.load(model_path)
else:
    # Load the model from the Whisper library
    model = whisper.load_model("small")
    # Save the model to a file
    torch.save(model, model_path)

# Transcribe the audio file
result = model.transcribe("audio.mp3")

# Print the transcribed text
print(result["text"])