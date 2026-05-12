import whisper

# Load Whisper once at startup
print("Loading Whisper model...")
model = whisper.load_model("small")
print("Whisper loaded.")


def transcribe(audio_path):
    """Convert audio file to text."""
    result = model.transcribe(audio_path)
    return result["text"].strip()