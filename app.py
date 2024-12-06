from flask import Flask, render_template, request
import os
from google.cloud import speech
import pyaudio
import wave
import threading

app = Flask(__name__)

# Path to your Google Cloud credentials JSON file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/adity/Downloads/speech-to-text-443918-bc8f683f7dd2.json"

# Global variables
recording = False
audio_filename = "recorded_audio.wav"

def record_audio():
    global recording
    p = pyaudio.PyAudio()

    # Open microphone stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)

    frames = []

    while recording:
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio to a file
    with wave.open(audio_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    print("Recording saved to", audio_filename)

def transcribe_audio():
    client = speech.SpeechClient()
    
    with open(audio_filename, "rb") as audio_file:
        audio = speech.RecognitionAudio(content=audio_file.read())
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    # Join the transcriptions from all results
    transcription = " ".join([result.alternatives[0].transcript for result in response.results])
    return transcription

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording
    if not recording:
        recording = True
        # Start recording in a new thread to avoid blocking the main thread
        threading.Thread(target=record_audio).start()
        return "Recording started"
    return "Already recording"

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    if recording:
        recording = False
        return "Recording stopped"
    return "No active recording"

@app.route('/transcribe', methods=['POST'])
def transcribe():
    transcription = transcribe_audio()
    return f"Transcription: {transcription}"

if __name__ == '__main__':
    app.run(debug=True)
