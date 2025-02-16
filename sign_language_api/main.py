from fastapi import FastAPI, WebSocket, HTTPException, Query
import uvicorn
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import json
import base64
from gtts import gTTS
import os
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكن تغييرها لاحقًا لتحديد النطاقات المسموحة
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load JSON files
def load_labels(language: str, type: str):
    file_path = f"labels/{language}_{type}.json"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# declare model , labels , language
models = {}
labels = {}
selected_language = None
text_buffer = ""

# extracte landmarks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

@app.get("/")
def home():
    return {"message": "Sign Language API is running"}
    
# set language 
@app.get("/set_language")
def set_language(language: str = Query(..., description="selecte language 'arabic' or 'english'")):
    global models, labels, selected_language, text_buffer

    if language not in ["arabic", "english"]:
        raise HTTPException(status_code=400, detail=" Language not supported! Choose 'arabic' or 'english'.")

    if selected_language == language:
        return {"message": f" {language} models are already loaded!"}

    # load models
    models = {
        "letters": tf.keras.models.load_model(f"models/{'Aalpha2' if language == 'arabic' else 'E_alpha'}_sign_language_model.h5"),
        "numbers": tf.keras.models.load_model(f"models/{'AN2' if language == 'arabic' else 'EN'}_sign_language_model.h5"),
    }

    # load labels
    labels = {
        "letters": load_labels(language, "letters"),
        "numbers": load_labels(language, "numbers"),
    }

    selected_language = language
    text_buffer = ""
    return {"message": f" {language} models loaded successfully!"}

# extract landmarks from image
def process_frame(frame):
    global text_buffer
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                        [lm.y for lm in hand_landmarks.landmark] + \
                        [lm.z for lm in hand_landmarks.landmark]

            landmarks_array = np.array(landmarks).reshape(1, -1)

            # prediction (letters + numbers)
            prediction_letters = models["letters"].predict(landmarks_array)
            prediction_numbers = models["numbers"].predict(landmarks_array)

            predicted_index_letters = np.argmax(prediction_letters)
            predicted_index_numbers = np.argmax(prediction_numbers)

            confidence_letters = np.max(prediction_letters)
            confidence_numbers = np.max(prediction_numbers)

            if confidence_letters > confidence_numbers:
                final_label = labels["letters"].get(str(predicted_index_letters), "Unknown")
                label_type = "letter"
            else:
                final_label = labels["numbers"].get(str(predicted_index_numbers), "Unknown")
                label_type = "number"

            # delete and space
            if selected_language == "arabic":
                if final_label == "حذف":
                    text_buffer = text_buffer[:-1]
                    return {"predicted_label": "حذف", "action": "delete_last_character"}
                elif final_label == "مسافة":
                    text_buffer += " "
                    return {"predicted_label": "مسافة", "action": "add_space"}
            else:
                if final_label == "delete":
                    text_buffer = text_buffer[:-1]
                    return {"predicted_label": "delete", "action": "delete_last_character"}
                elif final_label == "space":
                    text_buffer += " "
                    return {"predicted_label": "space", "action": "add_space"}
            
            text_buffer += final_label
            return {
                "predicted_label": final_label,
                "type": label_type,
                "confidence": max(confidence_letters, confidence_numbers),
                "text_buffer": text_buffer
            }
    
    return {"error": " No hand detected!"}


# WebSocket 
@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            np_arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": " Invalid frame received!"})
                continue

            prediction_result = process_frame(frame)
            await websocket.send_json(prediction_result)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

# text to voice
@app.post("/text_to_speech")
def text_to_speech(text: str):
    if not selected_language:
        raise HTTPException(status_code=400, detail="  use  not selected yet! /set_language selecte language")

    language_name = "Arabic" if selected_language == "arabic" else "English"
    
    
    os.makedirs("static", exist_ok=True)

    file_path = f"static/output_{language_name}.mp3"
    tts = gTTS(text=text, lang="ar" if selected_language == "arabic" else "en")
    tts.save(file_path)

    return {"message": " create voice", "file": file_path}

@app.post("/share")
def share(text: str, as_audio: bool = False):
    if not selected_language:
        raise HTTPException(status_code=400, detail="use  not selected yet! /set_language selecte language")

    language_name = "Arabic" if selected_language == "arabic" else "English"
    
    if as_audio:
        os.makedirs("static", exist_ok=True)  

        file_path = f"static/shared_audio_{language_name}.mp3"
        tts = gTTS(text=text, lang="ar" if selected_language == "arabic" else "en")
        tts.save(file_path)
        return {"message": "  voice done", "file": file_path}
    
    return {"message": "text done", "text": text}

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))  # استخدام المنفذ المتوفر من Railway
#     uvicorn.run(app, host="0.0.0.0", port=port)



