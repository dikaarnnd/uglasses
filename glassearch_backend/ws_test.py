import asyncio
import websockets

async def handler(websocket):
    print("✅ Client connected")
    try:
        async for message in websocket:
            # Terima data dari RN
            await websocket.send("Frame received")
    except websockets.exceptions.ConnectionClosed:
        print("❌ Client disconnected")

start_server = websockets.serve(handler, "0.0.0.0", 8000)

print("🚀 Server WebSocket murni berjalan di port 8000")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()