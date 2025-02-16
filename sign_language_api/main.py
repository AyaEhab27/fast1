from fastapi import FastAPI, WebSocket, HTTPException, Query
import uvicorn
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import uuid
from gtts import gTTS
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load JSON labels

def load_labels(language: str, type: str):
    file_path = f"labels/{language}_{type}.json"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Load models once
models = {
    "arabic": {
        "letters": tf.keras.models.load_model("models/Aalpha2_sign_language_model.h5"),
        "numbers": tf.keras.models.load_model("models/AN2_sign_language_model.h5"),
    },
    "english": {
        "letters": tf.keras.models.load_model("models/E_alpha_sign_language_model.h5"),
        "numbers": tf.keras.models.load_model("models/EN_sign_language_model.h5"),
    },
}

labels = {
    "arabic": {
        "letters": load_labels("arabic", "letters"),
        "numbers": load_labels("arabic", "numbers"),
    },
    "english": {
        "letters": load_labels("english", "letters"),
        "numbers": load_labels("english", "numbers"),
    },
}

selected_language = None
text_buffer = ""

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

@app.get("/set_language")
def set_language(language: str = Query(..., description="Choose 'arabic' or 'english'")):
    global selected_language, text_buffer
    if language not in ["arabic", "english"]:
        raise HTTPException(status_code=400, detail="Language not supported! Choose 'arabic' or 'english'.")
    selected_language = language
    text_buffer = ""
    return {"message": f"{language} language set successfully!"}

# Process frame & predict

def process_frame(frame):
    global text_buffer
    if not selected_language:
        return {"error": "Language not selected! Use /set_language"}

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                        [lm.y for lm in hand_landmarks.landmark] + \
                        [lm.z for lm in hand_landmarks.landmark]
            landmarks_array = np.array(landmarks).reshape(1, -1)

            # Predict letters & numbers
            prediction_letters = models[selected_language]["letters"].predict(landmarks_array)
            prediction_numbers = models[selected_language]["numbers"].predict(landmarks_array)

            predicted_index_letters = np.argmax(prediction_letters)
            predicted_index_numbers = np.argmax(prediction_numbers)

            confidence_letters = np.max(prediction_letters)
            confidence_numbers = np.max(prediction_numbers)
            
            threshold = 0.8  # Confidence threshold
            if confidence_letters > threshold or confidence_numbers > threshold:
                if confidence_letters > confidence_numbers:
                    final_label = labels[selected_language]["letters"].get(str(predicted_index_letters), "Unknown")
                    label_type = "letter"
                else:
                    final_label = labels[selected_language]["numbers"].get(str(predicted_index_numbers), "Unknown")
                    label_type = "number"
            else:
                return {"error": "Low confidence in prediction."}

            # Handle space & delete
            if selected_language == "arabic" and final_label in ["حذف", "مسافة"]:
                if final_label == "حذف":
                    text_buffer = text_buffer[:-1]
                else:
                    text_buffer += " "
                return {"predicted_label": final_label, "action": final_label}
            elif selected_language == "english" and final_label in ["delete", "space"]:
                if final_label == "delete":
                    text_buffer = text_buffer[:-1]
                else:
                    text_buffer += " "
                return {"predicted_label": final_label, "action": final_label}
            
            text_buffer += final_label
            return {"predicted_label": final_label, "type": label_type, "confidence": max(confidence_letters, confidence_numbers), "text_buffer": text_buffer}
    return {"error": "No hand detected!"}

# WebSocket Video Processing
@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            np_arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"error": "Invalid frame received!"})
                continue
            prediction_result = process_frame(frame)
            await websocket.send_json(prediction_result)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

# Text-to-Speech
@app.post("/text_to_speech")
def text_to_speech(text: str):
    if not selected_language:
        raise HTTPException(status_code=400, detail="Language not selected! Use /set_language")
    os.makedirs("static", exist_ok=True)
    file_name = f"static/{uuid.uuid4()}.mp3"
    tts = gTTS(text=text, lang="ar" if selected_language == "arabic" else "en")
    tts.save(file_name)
    return {"message": "Voice created", "file": file_name}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
