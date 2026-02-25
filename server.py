"""
VoiceAgent — Voice-First AI Agent Interface
Uses Moonshine STT for local, fast speech-to-text.
WebSocket server bridges browser mic → STT → AI agent → TTS response.
"""

import asyncio
import json
import os
import struct
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import Thread

import websockets
import moonshine_voice as mv

# Initialize Moonshine
assets_dir = os.path.dirname(str(mv.get_model_path('en')))
model_path = os.path.join(assets_dir, 'tiny-en')
SAMPLE_RATE = 16000


def transcribe_audio(audio_float32: list[float]) -> dict:
    """Transcribe audio using Moonshine STT."""
    t = mv.transcriber.Transcriber(model_path, model_arch=mv.ModelArch.TINY)
    start = time.time()
    result = t.transcribe_without_streaming(audio_float32)
    elapsed = time.time() - start
    t.close()

    # Parse result: "[0.10s] text\n[1.63s] text\n..."
    lines = []
    full_text = []
    for line in str(result).strip().split('\n'):
        line = line.strip()
        if line:
            # Extract text after timestamp
            if ']' in line:
                text = line.split(']', 1)[1].strip()
            else:
                text = line
            if text:
                lines.append(text)
                full_text.append(text)

    return {
        'text': ' '.join(full_text),
        'lines': lines,
        'duration_ms': round(elapsed * 1000),
        'audio_seconds': round(len(audio_float32) / SAMPLE_RATE, 2),
    }


async def handle_client(websocket):
    """Handle a WebSocket client connection."""
    print(f"Client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Binary: raw float32 PCM audio at 16kHz
                n_samples = len(message) // 4
                audio_data = list(struct.unpack(f'{n_samples}f', message))

                if n_samples < SAMPLE_RATE * 0.3:  # Less than 300ms
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Audio too short (< 300ms)'
                    }))
                    continue

                print(f"Received {n_samples} samples ({n_samples/SAMPLE_RATE:.1f}s)")
                result = transcribe_audio(audio_data)
                print(f"Transcribed in {result['duration_ms']}ms: {result['text']}")

                await websocket.send(json.dumps({
                    'type': 'transcription',
                    **result
                }))
            elif isinstance(message, str):
                data = json.loads(message)
                if data.get('type') == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))
    except websockets.exceptions.ConnectionClosed:
        pass
    print(f"Client disconnected: {websocket.remote_address}")


async def main():
    port = int(os.environ.get('PORT', 8765))
    print(f"VoiceAgent WebSocket server starting on ws://localhost:{port}")
    print(f"Using Moonshine tiny-en model from {model_path}")

    # Warm up the model
    print("Warming up model...")
    warmup = [0.0] * SAMPLE_RATE  # 1 second of silence
    t = mv.transcriber.Transcriber(model_path, model_arch=mv.ModelArch.TINY)
    t.transcribe_without_streaming(warmup)
    t.close()
    print("Model ready!")

    # Start HTTP server for the UI
    http_port = int(os.environ.get('HTTP_PORT', 8766))
    server_dir = os.path.dirname(os.path.abspath(__file__))

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=server_dir, **kwargs)
        def log_message(self, format, *args):
            pass  # Silence HTTP logs

    httpd = HTTPServer(('localhost', http_port), Handler)
    Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"UI available at http://localhost:{http_port}")

    async with websockets.serve(handle_client, "localhost", port):
        await asyncio.Future()


if __name__ == '__main__':
    asyncio.run(main())
