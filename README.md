# VoiceAgent — Voice-First AI Agent Interface

Talk to an AI agent using your voice. Local speech-to-text via [Moonshine](https://github.com/usefulmachines/moonshine), LLM responses via GPT-4o-mini, spoken back via browser TTS.

## Performance

- **STT:** ~200ms for 10s audio on Apple M4 Pro (49x real-time)
- **Full loop:** ~1.5s speak → transcribe → think → respond
- **Cost:** STT is 100% local/free. LLM is ~$0.001/conversation turn.

## Architecture

```
Browser Mic → WebSocket → Moonshine STT (local) → GPT-4o-mini → WebSocket → Browser TTS
```

Three modes:
1. **`agent.html`** — Full voice agent (STT → LLM → TTS) ← **start here**
2. **`index.html`** — Push-to-talk transcription only
3. **`streaming.html`** — Real-time streaming transcription

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Agent mode (needs OPENAI_API_KEY in keychain or env)
python agent_server.py
# Open http://localhost:8766/agent.html

# STT-only mode
python server.py
# Open http://localhost:8766

# Streaming mode
python server_streaming.py
# Open http://localhost:8766/streaming.html
```

## Requirements

- Python 3.11+
- macOS / Linux (Moonshine ONNX runs on CPU)
- Browser with mic access
- OpenAI API key (for agent mode only)

## How It Works

1. **Hold** the mic button (or spacebar) and speak
2. **Release** to send audio to Moonshine STT
3. Your words appear instantly (~200ms)
4. AI thinks and responds (~500-1500ms)
5. Response is spoken aloud via browser SpeechSynthesis
6. Full conversation history maintained for context

## Tech Stack

- **Moonshine** (tiny-en) — Local STT, ONNX runtime, ~50MB model
- **GPT-4o-mini** — Fast, cheap LLM for conversational responses
- **Browser SpeechSynthesis** — Zero-cost TTS, uses system voices
- **WebSocket** — Low-latency audio streaming
- **Vanilla HTML/CSS/JS** — No build step, no framework

## License

MIT
