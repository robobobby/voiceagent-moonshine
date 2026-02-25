"""
VoiceAgent — Full Agent Server
STT (Moonshine) → LLM (OpenAI) → TTS (OpenAI tts-1)
Push-to-talk: hold to speak, release to get AI response.
"""

import asyncio
import json
import os
import struct
import subprocess
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread

import websockets
import moonshine_voice as mv

# Moonshine STT setup
assets_dir = os.path.dirname(str(mv.get_model_path('en')))
model_path = os.path.join(assets_dir, 'tiny-en')
SAMPLE_RATE = 16000

# OpenAI setup
OPENAI_API_KEY = subprocess.run(
    ['security', 'find-generic-password', '-s', 'openai-api-key', '-w'],
    capture_output=True, text=True
).stdout.strip()

SYSTEM_PROMPT = """You are VoiceAgent, a helpful AI assistant optimized for voice conversation.
Keep responses concise and conversational — they'll be spoken aloud via text-to-speech.
Aim for 1-3 sentences unless the user asks for detail.
Be warm, direct, and natural. Avoid markdown formatting, bullet points, or code blocks.
If you need to give a list, say it naturally: "first... second... and third..."
"""

# Shared transcriber (warm, reused)
transcriber = None


def get_transcriber():
    global transcriber
    if transcriber is None:
        transcriber = mv.transcriber.Transcriber(model_path, model_arch=mv.ModelArch.TINY)
    return transcriber


def transcribe_audio(audio_float32: list[float]) -> dict:
    """Transcribe audio using Moonshine STT."""
    t = get_transcriber()
    start = time.time()
    result = t.transcribe_without_streaming(audio_float32)
    elapsed = time.time() - start

    lines = []
    for line in str(result).strip().split('\n'):
        line = line.strip()
        if line:
            text = line.split(']', 1)[1].strip() if ']' in line else line
            if text:
                lines.append(text)

    return {
        'text': ' '.join(lines),
        'duration_ms': round(elapsed * 1000),
        'audio_seconds': round(len(audio_float32) / SAMPLE_RATE, 2),
    }


async def get_llm_response(conversation: list[dict]) -> str:
    """Get LLM response from OpenAI (async via thread)."""
    import urllib.request

    def _call():
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation
        body = json.dumps({
            "model": "gpt-4o-mini",
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7,
        }).encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]

    return await asyncio.get_event_loop().run_in_executor(None, _call)


async def generate_tts(text: str, voice: str = "onyx") -> bytes | None:
    """Generate speech audio via OpenAI TTS API. Returns MP3 bytes."""
    import urllib.request

    def _call():
        body = json.dumps({
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "response_format": "mp3",
        }).encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/speech",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read()

    return await asyncio.get_event_loop().run_in_executor(None, _call)


async def handle_client(websocket):
    """Handle a WebSocket client — full voice agent loop."""
    print(f"Client connected: {websocket.remote_address}")
    conversation = []  # Conversation history for context

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Binary: raw float32 PCM audio at 16kHz
                n_samples = len(message) // 4
                audio_data = list(struct.unpack(f'{n_samples}f', message))

                if n_samples < SAMPLE_RATE * 0.3:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Audio too short (< 300ms)'
                    }))
                    continue

                print(f"Received {n_samples} samples ({n_samples/SAMPLE_RATE:.1f}s)")

                # Step 1: STT
                stt_result = transcribe_audio(audio_data)
                user_text = stt_result['text']
                print(f"STT ({stt_result['duration_ms']}ms): {user_text}")

                if not user_text.strip():
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'No speech detected'
                    }))
                    continue

                # Send transcription immediately
                await websocket.send(json.dumps({
                    'type': 'user_message',
                    'text': user_text,
                    'stt_ms': stt_result['duration_ms'],
                    'audio_seconds': stt_result['audio_seconds'],
                }))

                # Step 2: LLM
                conversation.append({"role": "user", "content": user_text})
                # Keep last 20 messages for context
                if len(conversation) > 20:
                    conversation = conversation[-20:]

                llm_start = time.time()
                try:
                    response_text = await get_llm_response(conversation)
                except Exception as e:
                    print(f"LLM error: {e}")
                    response_text = "Sorry, I couldn't process that. Could you try again?"

                llm_ms = round((time.time() - llm_start) * 1000)
                print(f"LLM ({llm_ms}ms): {response_text}")

                conversation.append({"role": "assistant", "content": response_text})

                # Step 3: Send text response immediately
                total_ms = stt_result['duration_ms'] + llm_ms
                await websocket.send(json.dumps({
                    'type': 'agent_response',
                    'text': response_text,
                    'llm_ms': llm_ms,
                    'total_ms': total_ms,
                }))

                # Step 4: Generate TTS audio via OpenAI and send as binary
                try:
                    tts_audio = await generate_tts(response_text)
                    if tts_audio:
                        # Send a JSON header first so client knows audio is coming
                        await websocket.send(json.dumps({
                            'type': 'tts_audio',
                            'format': 'mp3',
                            'size': len(tts_audio),
                        }))
                        await websocket.send(tts_audio)
                except Exception as e:
                    print(f"TTS error: {e}")

            elif isinstance(message, str):
                data = json.loads(message)
                if data.get('type') == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))
                elif data.get('type') == 'clear_history':
                    conversation.clear()
                    await websocket.send(json.dumps({'type': 'history_cleared'}))

    except websockets.exceptions.ConnectionClosed:
        pass
    print(f"Client disconnected: {websocket.remote_address}")


async def main():
    port = int(os.environ.get('PORT', 8765))
    http_port = int(os.environ.get('HTTP_PORT', 8766))

    print(f"VoiceAgent Server starting on ws://localhost:{port}")
    print(f"Model: Moonshine tiny-en + GPT-4o-mini")

    # Warm up STT
    print("Warming up Moonshine...")
    t = get_transcriber()
    t.transcribe_without_streaming([0.0] * SAMPLE_RATE)
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
    print(f"UI: http://localhost:{http_port}/agent.html")

    async with websockets.serve(handle_client, "localhost", port, max_size=10*1024*1024):
        await asyncio.Future()


if __name__ == '__main__':
    asyncio.run(main())
