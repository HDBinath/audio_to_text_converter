import atexit
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import riva.client
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
FUNCTION_ID = os.getenv("NVIDIA_ASR_FUNCTION_ID", "b702f636-f60c-4a3d-a6f4-f3568c13bd7d")
RIVA_URI = "grpc.nvcf.nvidia.com:443"

TARGET_SR = 16000
TARGET_SAMPLE_WIDTH = 2
TARGET_CHANNELS = 1

TEMP_ROOT = Path(tempfile.gettempdir()) / "nvidia_stt_app"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def _new_temp_dir() -> Path:
    d = TEMP_ROOT / uuid.uuid4().hex
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cleanup_dir(d: Optional[Path]) -> None:
    if d and d.exists():
        shutil.rmtree(d, ignore_errors=True)


def _cleanup_all_temp() -> None:
    shutil.rmtree(TEMP_ROOT, ignore_errors=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)


_cleanup_all_temp()
atexit.register(_cleanup_all_temp)


def _load_and_normalize(path: str) -> AudioSegment:
    try:
        audio = AudioSegment.from_file(path)
    except Exception as e:
        raise RuntimeError(
            f"Could not decode '{Path(path).name}'. Make sure ffmpeg is installed "
            f"and the file isn't corrupted. ({e})"
        )
    audio = (
        audio.set_frame_rate(TARGET_SR)
        .set_channels(TARGET_CHANNELS)
        .set_sample_width(TARGET_SAMPLE_WIDTH)
    )
    return audio


def _segment_to_pcm16_bytes(segment: AudioSegment) -> bytes:
    samples = np.array(segment.get_array_of_samples(), dtype=np.int16)
    return samples.tobytes()


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def transcribe(audio_path: Optional[str], progress: gr.Progress = gr.Progress()):
    if not NVIDIA_API_KEY:
        return "Error: NVIDIA_API_KEY not found. Add it to your .env file.", ""

    if not audio_path:
        return "Please upload or record audio first.", ""

    if not os.path.exists(audio_path):
        return "Audio file not found on disk.", ""

    work_dir = _new_temp_dir()
    try:
        progress(0.1, desc="Loading and normalizing audio...")
        audio = _load_and_normalize(audio_path)
        duration_str = _format_duration(len(audio) / 1000)

        progress(0.3, desc="Connecting to NVIDIA API...")
        auth = riva.client.Auth(
            uri=RIVA_URI,
            use_ssl=True,
            metadata_args=[
                ["function-id", FUNCTION_ID],
                ["authorization", f"Bearer {NVIDIA_API_KEY}"],
            ],
        )
        asr_service = riva.client.ASRService(auth)

        config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=TARGET_SR,
            language_code="en",
            max_alternatives=1,
            enable_automatic_punctuation=True,
            audio_channel_count=TARGET_CHANNELS,
        )

        pcm_bytes = _segment_to_pcm16_bytes(audio)

        progress(0.5, desc="Transcribing with whisper-large-v3...")
        response = asr_service.offline_recognize(pcm_bytes, config)

        pieces = []
        for result in response.results:
            if result.alternatives:
                pieces.append(result.alternatives[0].transcript.strip())

        progress(1.0, desc="Done.")
        full_text = " ".join(p for p in pieces if p)
        if not full_text:
            return "No speech detected in the audio.", duration_str
        return full_text, duration_str

    except RuntimeError as e:
        return str(e), ""
    except Exception as e:
        return f"Error: {e}", ""
    finally:
        _cleanup_dir(work_dir)


def clear_all():
    _cleanup_all_temp()
    return None, "", "", "Ready."


CUSTOM_CSS = """
#status_line { font-size: 0.9em; color: var(--body-text-color-subdued); }
"""

with gr.Blocks(title="NVIDIA Speech-to-Text") as demo:
    gr.Markdown("# NVIDIA Speech-to-Text")
    gr.Markdown(
        "Upload a file or record from your microphone. "
        "Supports **WAV, MP3, M4A, FLAC, OGG, WEBM** and most other common formats."
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                type="filepath",
                label="Upload or Record Audio",
                sources=["upload", "microphone"],
            )
            duration_box = gr.Textbox(label="Audio length", interactive=False)

            with gr.Row():
                transcribe_btn = gr.Button("Transcribe", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")

            status_line = gr.Markdown("Ready.", elem_id="status_line")

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="Transcript",
                placeholder="Transcription will appear here...",
                lines=16,
            )

    def _on_transcribe(audio_path):
        text, duration = transcribe(audio_path)
        status = "Transcription complete." if not text.startswith(("Error", "Please", "No speech")) else text
        return text, duration, status

    transcribe_btn.click(
        fn=_on_transcribe,
        inputs=audio_input,
        outputs=[output, duration_box, status_line],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[audio_input, output, duration_box, status_line],
    )

if __name__ == "__main__":
    try:
        demo.launch(css=CUSTOM_CSS, theme=gr.themes.Soft())
    finally:
        _cleanup_all_temp()
