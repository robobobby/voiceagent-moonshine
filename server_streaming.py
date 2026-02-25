"""
VoiceAgent — Streaming STT Server
Real-time transcription using Moonshine's streaming API.
Audio flows in continuously, partial transcripts flow back.
"""

import asyncio
import json
import os
import struct
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread

import websockets
import moonshine_voice as mv

assets_dir = os.path.dirname(str(mv.get_model_path('en')))
model_path = os.path.join(assets_dir, 'tiny-en')
SAMPLE_RATE = 16000
CHUNK_SIZE = 4096  # ~256ms at 16kHz


class StreamingSession:
    """Per-client streaming transcription session."""

    def __init__(self):
        self.transcriber = mv.transcriber.Transcriber(
            model_path,
            model_arch=mv.ModelArch.TINY,
            update_interval=0.3,
        )
        self.is_streaming = False
        self.partial_text = ""

    def start_stream(self):
        """Start a new streaming session."""
        self.is_streaming = True
        self.partial_text = ""
        self.transcriber.start()

    @staticmethod
    def _extract_text(raw: str) -> str:
        """Extract plain text from timestamped transcript lines."""
        lines = []
        for line in raw.strip().split('\n'):
            line = line.strip()
            if ']' in line:
                lines.append(line.split(']', 1)[1].strip())
            elif line:
                lines.append(line)
        return ' '.join(lines)

    def feed_audio(self, audio_float32: list[float]) -> str | None:
        """Feed audio chunk, return partial transcript if updated."""
        if not self.is_streaming:
            return None
        self.transcriber.add_audio(audio_float32)
        transcript = self.transcriber.update_transcription()
        if transcript:
            text = self._extract_text(str(transcript))
            if text and text != self.partial_text:
                self.partial_text = text
                return self.partial_text
        return None

    def stop_stream(self) -> str:
        """Stop streaming, return final transcript."""
        self.is_streaming = False
        self.transcriber.stop()
        final = self.partial_text
        self.partial_text = ""
        return final

    def close(self):
        self.transcriber.close()


async def handle_client(websocket):
    """Handle a WebSocket client with streaming transcription."""
    print(f"Client connected: {websocket.remote_address}")
    session = StreamingSession()

    try:
        async for message in websocket:
            if isinstance(message, str):
                data = json.loads(message)
                msg_type = data.get('type')

                if msg_type == 'start_stream':
                    session.start_stream()
                    await websocket.send(json.dumps({
                        'type': 'stream_started'
                    }))

                elif msg_type == 'stop_stream':
                    final = session.stop_stream()
                    await websocket.send(json.dumps({
                        'type': 'final_transcript',
                        'text': final,
                    }))

                elif msg_type == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))

            elif isinstance(message, bytes):
                # Binary: raw float32 PCM audio chunk
                n_samples = len(message) // 4
                audio_data = list(struct.unpack(f'{n_samples}f', message))

                partial = session.feed_audio(audio_data)
                if partial:
                    await websocket.send(json.dumps({
                        'type': 'partial_transcript',
                        'text': partial,
                    }))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        session.close()
    print(f"Client disconnected: {websocket.remote_address}")


async def main():
    port = int(os.environ.get('PORT', 8765))
    http_port = int(os.environ.get('HTTP_PORT', 8766))

    print(f"VoiceAgent Streaming Server starting on ws://localhost:{port}")
    print(f"Using Moonshine tiny-en model from {model_path}")

    # Warm up
    print("Warming up model...")
    t = mv.transcriber.Transcriber(model_path, model_arch=mv.ModelArch.TINY)
    t.transcribe_without_streaming([0.0] * SAMPLE_RATE)
    t.close()
    print("Model ready!")

    # HTTP server
    server_dir = os.path.dirname(os.path.abspath(__file__))

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=server_dir, **kwargs)
        def log_message(self, format, *args):
            pass

    httpd = HTTPServer(('localhost', http_port), Handler)
    Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"UI available at http://localhost:{http_port}")

    async with websockets.serve(handle_client, "localhost", port):
        await asyncio.Future()


if __name__ == '__main__':
    asyncio.run(main())
