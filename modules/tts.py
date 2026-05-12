import pyttsx3


def speak(text, output_path="output.wav"):
    """Convert text to speech and save as wav file."""
    engine = pyttsx3.init()
    engine.setProperty("rate", 200)  # speech speed
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    return output_path