import asyncio
import websockets
import cv2
import numpy as np

async def send_video_frame():
    uri = "ws://127.0.0.1:8000/ws/video"
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)  # تشغيل الكاميرا

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, buffer = cv2.imencode('.jpg', frame)
            await websocket.send(buffer.tobytes())

            response = await websocket.recv()
            print("Prediction:", response)

        cap.release()

asyncio.run(send_video_frame())
