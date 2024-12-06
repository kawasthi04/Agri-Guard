import pyaudio
import wave
from google.cloud import speech
import io

def record_audio(output_file, record_seconds=5, sample_rate=16000):
    """Records audio from the microphone and saves it to a .wav file."""
    chunk = 1024  # Record in chunks of 1024 samples
    format = pyaudio.paInt16  # 16-bit audio format
    channels = 1  # Mono audio
    sample_rate = sample_rate  # Sampling rate

    p = pyaudio.PyAudio()
    print("Recording...")

    # Open the microphone stream
    stream = p.open(format=format, channels=channels,
                    rate=sample_rate, input=True,
                    frames_per_buffer=chunk)

    frames = []

    # Record for the specified number of seconds
    for _ in range(0, int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording complete.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the audio to a file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

def transcribe_audio(audio_file_path):
    """Transcribes the audio file using Google Speech-to-Text API."""
    client = speech.SpeechClient()

    # Load the audio file
    with io.open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    # Configure the request
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    # Perform the transcription
    response = client.recognize(config=config, audio=audio)

    # Display the results
    for result in response.results:
        print(f"Transcript: {result.alternatives[0].transcript}")

# Main program
if __name__ == "__main__":
    audio_file = "recorded_audio.wav"

    # Step 1: Record Audio
    record_audio(audio_file, record_seconds=5)  # Record for 5 seconds

    # Step 2: Transcribe Audio
    transcribe_audio(audio_file)
