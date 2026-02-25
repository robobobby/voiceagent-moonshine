"""Single-port HTTPS + WSS server for VoiceAgent using aiohttp."""
import ssl, os, asyncio, json, struct, time
import aiohttp
from aiohttp import web

DIR = os.path.dirname(os.path.abspath(__file__))
CERT = os.path.join(DIR, 'cert.pem')
KEY = os.path.join(DIR, 'key.pem')

import sys
sys.path.insert(0, DIR)
from agent_server import get_transcriber, transcribe_audio, get_llm_response, SAMPLE_RATE


async def websocket_handler(request):
    ws = web.WebSocketResponse(max_msg_size=10*1024*1024)
    await ws.prepare(request)
    print(f"Client connected: {request.remote}", flush=True)

    conversation = []

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.BINARY:
            data = msg.data
            n_samples = len(data) // 4
            audio_data = list(struct.unpack(f'{n_samples}f', data))

            if n_samples < SAMPLE_RATE * 0.3:
                await ws.send_json({'type': 'error', 'message': 'Audio too short (< 300ms)'})
                continue

            print(f"Received {n_samples} samples ({n_samples/SAMPLE_RATE:.1f}s)", flush=True)

            # STT
            stt_result = transcribe_audio(audio_data)
            user_text = stt_result['text']
            print(f"STT ({stt_result['duration_ms']}ms): {user_text}", flush=True)

            if not user_text.strip():
                await ws.send_json({'type': 'error', 'message': 'No speech detected'})
                continue

            await ws.send_json({
                'type': 'user_message',
                'text': user_text,
                'stt_ms': stt_result['duration_ms'],
                'audio_seconds': stt_result['audio_seconds'],
            })

            # LLM
            conversation.append({"role": "user", "content": user_text})
            if len(conversation) > 20:
                conversation = conversation[-20:]

            llm_start = time.time()
            try:
                response_text = await get_llm_response(conversation)
            except Exception as e:
                print(f"LLM error: {e}", flush=True)
                response_text = "Sorry, I couldn't process that. Could you try again?"

            llm_ms = round((time.time() - llm_start) * 1000)
            print(f"LLM ({llm_ms}ms): {response_text}", flush=True)

            conversation.append({"role": "assistant", "content": response_text})

            await ws.send_json({
                'type': 'agent_response',
                'text': response_text,
                'llm_ms': llm_ms,
                'total_ms': stt_result['duration_ms'] + llm_ms,
            })

        elif msg.type == aiohttp.WSMsgType.TEXT:
            data = json.loads(msg.data)
            if data.get('type') == 'ping':
                await ws.send_json({'type': 'pong'})
            elif data.get('type') == 'clear_history':
                conversation.clear()
                await ws.send_json({'type': 'history_cleared'})

    print(f"Client disconnected: {request.remote}", flush=True)
    return ws


async def main():
    host = '0.0.0.0'
    port = 8766

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(CERT, KEY)

    print("Warming up Moonshine...", flush=True)
    t = get_transcriber()
    t.transcribe_without_streaming([0.0] * 16000)
    print("Model ready!", flush=True)

    app = web.Application()
    app.router.add_get('/ws', websocket_handler)
    app.router.add_static('/', DIR, show_index=True)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port, ssl_context=ssl_ctx)
    await site.start()

    print(f"VoiceAgent: https://agents-mac-mini.tail826413.ts.net:{port}/agent.html", flush=True)
    print(f"WebSocket:  wss://agents-mac-mini.tail826413.ts.net:{port}/ws", flush=True)
    print("Single port — one cert to accept!", flush=True)

    await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())
