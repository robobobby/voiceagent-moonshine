#!/usr/bin/env python3
"""
VoiceAgent Demo — Run the full pipeline without a microphone.
Uses bundled Moonshine test audio to demonstrate STT → LLM → TTS.
"""

import asyncio
import json
import struct
import time
import wave

import agent_server


DEMO_FILES = [
    ('.venv/lib/python3.14/site-packages/moonshine_voice/assets/beckett.wav', 'Samuel Beckett quote'),
    ('.venv/lib/python3.14/site-packages/moonshine_voice/assets/two_cities.wav', 'Tale of Two Cities'),
]


def load_wav(path: str) -> list[float]:
    with wave.open(path, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        samples = struct.unpack(f'{len(frames)//2}h', frames)
        return [s / 32768.0 for s in samples]


async def main():
    print("=" * 60)
    print("VoiceAgent Demo — Full Pipeline Test")
    print("=" * 60)
    print()

    for path, label in DEMO_FILES:
        print(f"--- {label} ---")
        try:
            audio = load_wav(path)
        except FileNotFoundError:
            print(f"  [skip] {path} not found")
            continue

        # STT
        stt_start = time.time()
        result = agent_server.transcribe_audio(audio)
        stt_ms = round((time.time() - stt_start) * 1000)

        print(f"  Audio: {result['audio_seconds']}s")
        print(f"  STT ({stt_ms}ms): \"{result['text']}\"")

        if not result['text'].strip():
            print("  [skip] No speech detected")
            continue

        # LLM
        llm_start = time.time()
        response = await agent_server.get_llm_response([
            {"role": "user", "content": result['text']}
        ])
        llm_ms = round((time.time() - llm_start) * 1000)

        print(f"  LLM ({llm_ms}ms): \"{response}\"")
        print(f"  Total: {stt_ms + llm_ms}ms")
        print()

    print("=" * 60)
    print("Demo complete. Run `python agent_server.py` for the full UI.")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
