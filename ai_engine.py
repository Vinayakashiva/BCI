import json
import pyttsx3
import pyaudio
import wave
from gtts import gTTS
# -------------------------
# CONFIG
# -------------------------
CONFIDENCE_THRESHOLD = 0.5

# -------------------------
# GENERATE SENTENCE
# -------------------------
def generate_sentence(anchor, emotion):

    a = anchor.lower()
    e = emotion.lower()

    if a == "thankyou":
        return "Thank you so much."
    elif a == "hello":
        return "Hello, how are you?"
    elif a == "helpme":
        if e == "distressed":
            return "Please help me urgently!"
        return "Please help me."
    elif a == "stop":
        if e == "angry":
            return "Stop right now!"
        return "Please stop."
    elif a == "yes":
        return "Yes, that's correct."

    return f"I want {anchor}."


# -------------------------
# TEXT → AUDIO FILE
# -------------------------
def text_to_wav(text, filename="output.wav"):

    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()


# -------------------------
# PLAY AUDIO USING PYAUDIO
# -------------------------
def play_audio(filename="output.wav"):

    wf = wave.open(filename, 'rb')

    p = pyaudio.PyAudio()

    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )

    data = wf.readframes(1024)

    while data:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()
    p.terminate()


# -------------------------
# MAIN PROCESS
# -------------------------
def process_json(file_path):

    with open(file_path, "r") as f:
        data = json.load(f)

    print("\n📥 Input JSON:")
    print(data)

    anchor = data["anchor_word"]
    emotion = data.get("emotion", "neutral")
    confidence = data.get("confidence", 1.0)

    if confidence < CONFIDENCE_THRESHOLD:
        text = "Uncertain prediction"
    else:
        text = generate_sentence(anchor, emotion)

    return text

def process_prediction(anchor, emotion="neutral", confidence=1.0):

    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain prediction"

    return generate_sentence(anchor, emotion)
# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":

    # 1. Read JSON + generate text
    output_text = process_json("input.json")

    print("\n🧾 Generated Sentence:")
    print(output_text)

    # 2. Convert text → audio file
    print("\n🎧 Generating audio file...")
    text_to_wav(output_text)

    # 3. Play using PyAudio
    print("🔊 Playing audio using PyAudio...")
    play_audio()