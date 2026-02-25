# VoiceAgent 🎙️

**Voice-first AI agent interface using [Moonshine STT](https://github.com/moonshine-ai/moonshine) for local, real-time speech-to-text.**

Zero cloud costs. Runs entirely on your machine. ~225ms transcription for 10s of audio on Apple Silicon.

## Features

- **Push-to-talk mode** (`index.html`) — Hold mic button, speak, release → instant transcription
- **Streaming mode** (`streaming.html`) — Click to start, words appear in real-time as you speak
- **Audio visualizer** — Live frequency display
- **Keyboard shortcut** — Hold/press Space to record
- **Touch support** — Works on mobile browsers

## Quick Start

```bash
# Create venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install moonshine-voice websockets

# Start the server
python3 server.py          # Push-to-talk mode
# OR
python3 server_streaming.py  # Streaming mode

# Open browser
open http://localhost:8766              # Push-to-talk UI
open http://localhost:8766/streaming.html  # Streaming UI
```

## Architecture

```
Browser (mic) → WebSocket (float32 PCM @ 16kHz) → Python server → Moonshine STT → WebSocket → Browser
```

- **`server.py`** — Push-to-talk: receives complete audio, transcribes in one shot
- **`server_streaming.py`** — Streaming: receives audio chunks continuously, returns partial transcripts
- **`index.html`** — Push-to-talk UI with frequency visualizer
- **`streaming.html`** — Streaming UI with live text display

## Model

Uses Moonshine `tiny-en` model (bundled with `moonshine-voice` package). Runs on CPU — no GPU required.

- **Inference speed:** ~225ms for 10s audio on M4 Pro
- **Streaming latency:** Partial transcripts every ~300ms
- **Languages:** en, ar, es, ja, ko, vi, uk, zh

## Requirements

- Python 3.10+
- macOS / Linux (tested on macOS arm64)
- A microphone
- A modern browser with WebRTC support

## License

MIT
