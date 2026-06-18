import os
import wave
from pathlib import Path

import gradio as gr
import numpy as np
import riva.client
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
FUNCTION_ID = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"


def _get_asr_service():
    auth = riva.client.Auth(
        uri="grpc.nvcf.nvidia.com:443",
        use_ssl=True,
        metadata_args=[
            ["function-id", FUNCTION_ID],
            ["authorization", f"Bearer {NVIDIA_API_KEY}"],
        ],
    )
    return riva.client.ASRService(auth)


def _load_wav(path: str) -> tuple[int, np.ndarray]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    dtype = np.int16 if sampwidth == 2 else np.int32
    data = np.frombuffer(frames, dtype=dtype)
    if nchannels > 1:
        data = data.reshape(-1, nchannels).mean(axis=1).astype(dtype)
    return sr, data


def _try_load_audio(path: str) -> tuple[int, np.ndarray]:
    try:
        return _load_wav(path)
    except wave.Error:
        pass
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        return audio.frame_rate, np.array(audio.get_array_of_samples(), dtype=np.int16)
    except Exception:
        raise RuntimeError(
            f"Could not decode '{Path(path).name}'. "
            "Install ffmpeg for non-WAV support, or upload a 16-bit mono WAV file."
        )


def transcribe(audio):
    if not NVIDIA_API_KEY:
        return "Error: NVIDIA_API_KEY not found in .env file"

    if audio is None:
        return "Please upload or record audio"

    if isinstance(audio, dict):
        path = audio.get("path")
    elif isinstance(audio, str):
        path = audio
    elif isinstance(audio, tuple):
        sample_rate, data = audio
        return _transcribe_audio(sample_rate, data)
    else:
        return f"Unexpected input type: {type(audio).__name__}"

    if not path or not os.path.exists(path):
        return "Audio file not found"

    try:
        sample_rate, data = _try_load_audio(path)
    except RuntimeError as e:
        return str(e)

    return _transcribe_audio(sample_rate, data)


def _transcribe_audio(sample_rate: int, data: np.ndarray) -> str:
    if data.dtype != np.int16:
        data = (data * 32767).astype(np.int16)

    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.int16)

    audio_bytes = data.tobytes()

    try:
        asr_service = _get_asr_service()
        config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=sample_rate,
            language_code="en",
            max_alternatives=1,
        )
        response = asr_service.offline_recognize(audio_bytes, config)

        if response.results:
            return response.results[0].alternatives[0].transcript
        return "No speech detected in the audio."

    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="NVIDIA Speech-to-Text") as demo:
    gr.Markdown("# NVIDIA Speech-to-Text Converter")
    gr.Markdown(
        "Upload an audio file or record from your microphone to get a transcript using NVIDIA's ASR API."
    )

    with gr.Row():
        audio_input = gr.Audio(
            type="filepath",
            label="Upload or Record Audio",
            sources=["upload", "microphone"],
        )

    transcribe_btn = gr.Button("Transcribe", variant="primary")

    output = gr.Textbox(
        label="Transcript",
        placeholder="Transcription will appear here...",
        lines=10,
    )

    transcribe_btn.click(fn=transcribe, inputs=audio_input, outputs=output)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
