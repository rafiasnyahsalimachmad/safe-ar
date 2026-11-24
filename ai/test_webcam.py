
import sys
import json
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

# ==========================================
# 1. PATH DAN KONFIGURASI
# ==========================================

AI_DIR = Path(__file__).resolve().parent

MODEL_PATH = AI_DIR / "models" / "gesture_classifier.joblib"
TASK_PATH = AI_DIR / "models" / "hand_landmarker.task"
SCHEMA_PATH = AI_DIR / "dataset" / "processed" / "feature_schema.json"

# Lokasi feature_extractor.py
RESULTS_DIR = AI_DIR / "results"
sys.path.insert(0, str(RESULTS_DIR))

from feature_extractor import normalize_landmarks

MIN_CONFIDENCE = 0.60
CAMERA_INDEX = 0

# ==========================================
# 2. LOAD MODEL
# ==========================================

print("Memuat model Random Forest...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

if not TASK_PATH.exists():
    raise FileNotFoundError(f"MediaPipe model tidak ditemukan: {TASK_PATH}")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_order = bundle["feature_order"]
class_names = bundle["class_names"]

with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
    schema = json.load(file)

if feature_order != schema["feature_order"]:
    raise ValueError("Urutan fitur model berbeda dengan feature_schema.json!")

if len(feature_order) != 63:
    raise ValueError("Model harus menggunakan 63 fitur.")

print("Model berhasil dimuat!")
print("Kelas:", class_names)

# ==========================================
# 3. INISIALISASI MEDIAPIPE
# ==========================================

print("Menginisialisasi MediaPipe...")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(TASK_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)

detector = HandLandmarker.create_from_options(options)

# ==========================================
# 4. INISIALISASI WEBCAM
# ==========================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    detector.close()
    raise RuntimeError(
        "Webcam tidak dapat dibuka. Periksa kamera atau CAMERA_INDEX."
    )

print("\nWebcam aktif!")
print("Tekan Q untuk keluar.")

# ==========================================
# 5. REAL-TIME GESTURE DETECTION
# ==========================================

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca frame webcam.")
            break

        height, width = frame.shape[:2]

        # Gunakan frame asli tanpa mirror untuk MediaPipe.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = detector.detect(mp_image)

        # ==================================
        # 6. CEK TANGAN
        # ==================================

        if not result.hand_landmarks:

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        else:

            for index, landmarks in enumerate(
                result.hand_landmarks
            ):

                # Ambil handedness
                handedness_result = result.handedness[index]

                if not handedness_result:
                    continue

                handedness = (
                    handedness_result[0].category_name
                )

                # Ambil 21 landmark x, y, z
                points = np.array(
                    [
                        [lm.x, lm.y, lm.z]
                        for lm in landmarks
                    ],
                    dtype=np.float64
                )

                # ==================================
                # 7. NORMALISASI 63 FITUR
                # ==================================

                try:

                    normalized, palm_scale = normalize_landmarks(
                        points,
                        image_size=(width, height),
                        handedness=handedness
                    )

                    features = normalized.flatten()

                    if len(features) != len(feature_order):
                        raise ValueError(
                            "Jumlah fitur tidak sesuai."
                        )

                except ValueError as error:

                    print("Normalisasi gagal:", error)
                    continue

                # ==================================
                # 8. PREDIKSI RANDOM FOREST
                # ==================================

                input_data = pd.DataFrame(
                    [features],
                    columns=feature_order
                )

                probabilities = model.predict_proba(
                    input_data
                )[0]

                best_index = int(np.argmax(probabilities))

                predicted_label = model.classes_[best_index]

                confidence = float(
                    probabilities[best_index]
                )

                if confidence >= MIN_CONFIDENCE:
                    display_label = predicted_label
                    status = "RECOGNIZED"
                    color = (0, 255, 0)
                else:
                    display_label = "UNCERTAIN"
                    status = "LOW CONFIDENCE"
                    color = (0, 165, 255)

                # ==================================
                # 9. GAMBAR LANDMARK
                # ==================================

                for lm in landmarks:

                    px = int(lm.x * width)
                    py = int(lm.y * height)

                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(
                            frame,
                            (px, py),
                            4,
                            (0, 255, 255),
                            -1
                        )

                # ==================================
                # 10. TAMPILKAN INFORMASI
                # ==================================

                text_y = 35 + (index * 90)

                cv2.putText(
                    frame,
                    f"Gesture: {display_label}",
                    (20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Confidence: {confidence * 100:.2f}%",
                    (20, text_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Hand: {handedness} | {status}",
                    (20, text_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

        # ==================================
        # 11. TAMPILKAN WEBCAM
        # ==================================

        cv2.imshow(
            "S.A.F.E. AR - Gesture Detection",
            frame
        )

        # Tekan Q untuk keluar
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    print("\nWebcam ditutup.")
