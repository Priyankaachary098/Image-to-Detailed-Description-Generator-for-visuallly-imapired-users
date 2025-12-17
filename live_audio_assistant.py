import os
import time
import tempfile

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound
import gradio as gr

# =========================
# LOAD MODEL
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Using device:", DEVICE)

DANGER_CLASSES = {
    # Vehicles (highest danger)
    "car": 3, "truck": 3, "bus": 3, "motorbike": 3, "bicycle": 2,
    "train": 3, "boat": 3, "airplane": 3,

    # Humans & animals
    "person": 2,
    "dog": 2, "cat": 2, "cow": 3, "horse": 3, "sheep": 2,

    # Furniture / obstacles
    "chair": 2, "sofa": 2, "bed": 2, "bench": 2,
    "table": 2, "dining table": 2, "desk": 2,

    # Elevation / falls
    "stairs": 3, "ladder": 3, "step": 3,

    # fallback
    "unknown": 2,
}

# Extra: list of vehicle-type classes for special logic
VEHICLE_CLASSES = {
    "car", "truck", "bus", "motorbike", "bicycle", "train", "boat", "airplane"
}

NEAR_AREA = 0.12  # base "near" threshold (for people / furniture etc.)

print("[INFO] Loading YOLO...")
model = YOLO("yolov8n.pt")
print("[INFO] YOLO loaded.")

# Warmup so first real frame is faster
print("[INFO] Warming up model...")
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
_ = model(dummy, verbose=False)
print("[INFO] Warmup done.")

# =========================
# GLOBALS FOR AUDIO
# =========================
last_spoken_text = ""
last_spoken_time = 0.0
audio_enabled = False   # toggled by Start/Stop buttons


# =========================
# SIMPLE TTS SPEAK FUNCTION
# =========================
def speak(text: str, language: str):
    if not text:
        return

    lang_codes = {
        "English": "en",
        "Hindi": "hi",
        "Kannada": "kn",
    }
    lang = lang_codes.get(language, "en")

    try:
        print(f"[TTS] Speaking in {language} ({lang}):", text[:80], "...")
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            path = fp.name
        tts.save(path)
        playsound(path)
        os.remove(path)
    except Exception as e:
        print("[TTS ERROR]", e)


# =========================
# MULTI-LANGUAGE SPEECH TEXT
# =========================
def build_speech_text(level: str, counts: dict, language: str) -> str:
    """
    counts = {"ahead": n1, "left": n2, "right": n3}
    Make sentences clearly different per language (no English mixing).
    """
    ahead = counts.get("ahead", 0)
    left = counts.get("left", 0)
    right = counts.get("right", 0)

    # Simple obstacle description (numbers only)
    if ahead == left == right == 0:
        obs_en = "no major obstacles near you"
    else:
        pieces = []
        if ahead > 0:
            pieces.append(f"{ahead} obstacle(s) in front")
        if left > 0:
            pieces.append(f"{left} obstacle(s) on the left")
        if right > 0:
            pieces.append(f"{right} obstacle(s) on the right")
        obs_en = ", ".join(pieces)

    if language == "English":
        if level == "SAFE":
            return f"Status safe. Path is clear, you can move forward slowly. I see {obs_en}."
        elif level == "CAUTION":
            return f"Status caution. Move carefully, there are some obstacles. I see {obs_en}."
        else:
            return f"Status danger. Stop and check the surroundings before moving. I see {obs_en}."

    if language == "Hindi":
        if level == "SAFE":
            base = "स्थिति सुरक्षित है. रास्ता साफ है, आप धीरे-धीरे आगे बढ़ सकते हैं."
        elif level == "CAUTION":
            base = "स्थिति सावधानी की है. धीरे चलिए, आसपास कुछ बाधाएँ हैं."
        else:
            base = "स्थिति खतरे की है. कृपया रुकिए और आसपास की स्थिति जाँचिए."
        return f"{base} आपके आसपास {obs_en} हैं."

    if language == "Kannada":
        if level == "SAFE":
            base = "ಸ್ಥಿತಿ ಸುರಕ್ಷಿತವಾಗಿದೆ. ದಾರಿ ಸ್ವಚ್ಛವಾಗಿದೆ, ನೀವು ನಿಧಾನವಾಗಿ ಮುಂದೆ ಹೋಗಬಹುದು."
        elif level == "CAUTION":
            base = "ಸ್ಥಿತಿ ಎಚ್ಚರಿಕೆಯಾಗಿದೆ. ನಿಧಾನವಾಗಿ ನಡೆಯಿರಿ, ಸುತ್ತಮುತ್ತ ಕೆಲವು ಅಡ್ಡಿಗಳು ಇವೆ."
        else:
            base = "ಸ್ಥಿತಿ ಅಪಾಯಕರವಾಗಿದೆ. ದಯವಿಟ್ಟು ನಿಲ್ಲಿ ಮತ್ತು ಸುತ್ತಮುತ್ತ ಪರಿಶೀಲಿಸಿ."
        return f"{base} ನಿಮ್ಮ ಹತ್ತಿರ {obs_en} ಇವೆ."

    # fallback
    return f"Status {level}. I see {obs_en}."


# =========================
# DANGER ANALYSIS (COUNTS + SPECIAL VEHICLE LOGIC)
# =========================
def analyze(frame, dets):
    h, w, _ = frame.shape
    frame_area = h * w

    danger_score = 0
    left_free = True
    right_free = True

    ahead_count = 0
    left_count = 0
    right_count = 0

    person_like_count = 0
    has_strong_hazard = False  # vehicles, stairs, etc.

    vehicles_ahead = 0
    big_vehicle_ahead = False  # large car/bus in front

    for d in dets:
        name = d["name"]
        x1, y1, x2, y2 = d["box"]
        area = (x2 - x1) * (y2 - y1)
        cx = (x1 + x2) / 2

        area_ratio = area / frame_area

        is_vehicle = name in VEHICLE_CLASSES

        if name in ["person", "dog", "cat", "cow", "horse", "sheep"]:
            person_like_count += 1

        severity = DANGER_CLASSES.get(name, DANGER_CLASSES["unknown"])

        # Vehicles and stairs etc. are considered strong hazards
        if severity >= 3 and name not in ["person"]:
            has_strong_hazard = True

        # region split + counts
        if cx < w / 3:
            region = "left"
            left_count += 1
        elif cx > 2 * w / 3:
            region = "right"
            right_count += 1
        else:
            region = "center"
            ahead_count += 1
            if is_vehicle:
                vehicles_ahead += 1
                if area_ratio > 0.06:  # a bit smaller threshold for vehicles
                    big_vehicle_ahead = True

        # danger weighting:
        # - for vehicles: slightly more sensitive
        # - for others: use NEAR_AREA
        if is_vehicle:
            if area_ratio > 0.06:
                # vehicles: more sensitive threshold
                if region == "center":
                    danger_score += 3 * severity
                else:
                    danger_score += 2 * severity
        else:
            if area_ratio > NEAR_AREA:
                if region == "center":
                    danger_score += 2 * severity
                else:
                    danger_score += severity

        # free space check
        if region == "left" and area_ratio > 0.05:
            left_free = False
        if region == "right" and area_ratio > 0.05:
            right_free = False

    # Base label from score
    if danger_score >= 8:
        level = "DANGER"
    elif danger_score >= 3:
        level = "CAUTION"
    else:
        level = "SAFE"

    # Extra safety rules for vehicles:
    # 1) Any big vehicle ahead => force DANGER
    if big_vehicle_ahead:
        level = "DANGER"

    # 2) Any vehicles ahead at all => at least CAUTION
    elif vehicles_ahead > 0 and level == "SAFE":
        level = "CAUTION"

    # Soften when only people / animals and no strong hazard
    if not has_strong_hazard and vehicles_ahead == 0:
        if level == "DANGER":
            level = "CAUTION"
        if person_like_count <= 1 and danger_score < 4:
            level = "SAFE"

    # English advice for UI text only
    if level == "SAFE":
        advice = "Path is clear. You can move forward."
    elif level == "CAUTION":
        if left_free and not right_free:
            advice = "Be careful. Move slightly left."
        elif right_free and not left_free:
            advice = "Be careful. Move slightly right."
        elif left_free and right_free:
            advice = "Move slowly, there are some obstacles around."
        else:
            advice = "Slow down, objects are close on all sides."
    else:  # DANGER
        if left_free and not right_free:
            advice = "Danger ahead. Move carefully to your left."
        elif right_free and not left_free:
            advice = "Danger ahead. Move carefully to your right."
        elif left_free and right_free:
            advice = "Danger ahead. Stop and choose a safe side to move."
        else:
            advice = "Danger ahead. Path is blocked."

    ui_objects = f"Obstacles ⇒ Ahead: {ahead_count} (vehicles: {vehicles_ahead}), Left: {left_count}, Right: {right_count}"
    counts = {"ahead": ahead_count, "left": left_count, "right": right_count}
    return level, advice, ui_objects, counts


# =========================
# LIVE PROCESSING FUNCTION
# =========================
def process_live(img, language):
    global last_spoken_text, last_spoken_time

    if img is None:
        return "Waiting...", "Waiting...", "No data"

    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w, _ = frame.shape
    frame_area = h * w

    # YOLO inference with good resolution
    res = model.predict(
        frame,
        verbose=False,
        imgsz=640,
        conf=0.5
    )[0]

    dets = []
    for box in res.boxes:
        cls = int(box.cls[0])
        raw_name = model.names[cls]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # geometry (for possible heuristics)
        box_w = x2 - x1
        box_h = y2 - y1
        area_ratio = (box_w * box_h) / frame_area
        cx = (x1 + x2) / 2

        name = raw_name

        # If YOLO calls a huge central thing something weird, treat as person
        if raw_name in ["vase", "potted plant", "bottle", "elephant", "bear", "giraffe", "zebra"]:
            if area_ratio > 0.18 and (w / 4) < cx < (3 * w / 4):
                name = "person"

        dets.append({"name": name, "box": (int(x1), int(y1), int(x2), int(y2))})

    level, advice_en, ui_objects, counts = analyze(frame, dets)

    # Build sentence in the selected language
    speech_text = build_speech_text(level, counts, language)

    # Speak only if enabled and changed
    if audio_enabled and speech_text:
        now = time.time()
        if speech_text != last_spoken_text and (now - last_spoken_time) > 2:
            speak(speech_text, language)
            last_spoken_text = speech_text
            last_spoken_time = now

    # UI always shows English text (for judges)
    return level, advice_en, ui_objects


# =========================
# AUDIO CONTROL BUTTONS
# =========================
def start_audio():
    global audio_enabled
    audio_enabled = True
    return "Voice guidance turned ON."


def stop_audio():
    global audio_enabled
    audio_enabled = False
    return "Voice guidance turned OFF."


# =========================
# GRADIO UI (Gradio 3.41.2 style)
# =========================
with gr.Blocks() as app:
    gr.Markdown(
        """
        # 🔊 VisionAid – Live Audio Assistant  
        Real-time danger detection + continuous audio guidance  
        **Speech Languages:** English / Hindi / Kannada
        """
    )

    with gr.Row():
        cam = gr.Image(
            source="webcam",      # Gradio 3.x: use source="webcam"
            streaming=True,
            type="numpy",
            label="Live Camera"
        )

        with gr.Column():
            danger = gr.Label(label="Danger Level")
            advice = gr.Textbox(label="Advice (English)", lines=2)
            objects = gr.Textbox(label="Obstacle Summary", lines=2)

            lang = gr.Radio(
                ["English", "Hindi", "Kannada"],
                value="English",
                label="Audio Language"
            )

            start_btn = gr.Button("▶ Start Audio Guidance")
            stop_btn = gr.Button("⏹ Stop Audio")

            msg = gr.Textbox(label="Status")

    # In Gradio 3.41.2, streaming=True causes .change to be called repeatedly
    cam.change(
        fn=process_live,
        inputs=[cam, lang],
        outputs=[danger, advice, objects],
    )

    start_btn.click(fn=start_audio, outputs=[msg])
    stop_btn.click(fn=stop_audio, outputs=[msg])

app.launch()
