import ctypes
import os
import pickle
import sys
import time
from typing import List, Optional, Sequence, Tuple

if os.name == "nt":
    _original_cdll = ctypes.CDLL

    class _CDLLProxy:
        def __init__(self, library):
            self._library = library
            self._ucrt = None

        def __getattr__(self, name):
            if name == "free":
                if self._ucrt is None:
                    self._ucrt = _original_cdll("ucrtbase.dll")
                return getattr(self._ucrt, "free")
            return getattr(self._library, name)

        def __setattr__(self, name, value):
            if name in {"_library", "_ucrt"}:
                object.__setattr__(self, name, value)
            else:
                setattr(self._library, name, value)

    def _patched_cdll(path, *args, **kwargs):
        library = _original_cdll(path, *args, **kwargs)
        return _CDLLProxy(library)

    ctypes.CDLL = _patched_cdll

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

# Core runtime constants.
MODEL_PATH = os.path.join("model", "sign_model.h5")
ENCODER_PATH = os.path.join("model", "label_encoder.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")
LANDMARKER_PATH = "hand_landmarker.task"
HOLD_DURATION = 1.5
CONFIDENCE_THRESHOLD = 0.85
CAM_INDEX = 0
PREDICTION_STRIDE = 2

# Hand skeleton connections used for manual drawing.
HAND_CONNECTIONS: Sequence[Tuple[int, int]] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def startup_checks() -> bool:
    # Validate required runtime files before opening the webcam.
    try:
        if not os.path.isfile(LANDMARKER_PATH):
            print(
                "Download: curl -o hand_landmarker.task -L "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            )
            return False

        if not os.path.isfile(MODEL_PATH):
            print("Run python train_model.py first")
            return False

        if not os.path.isfile(ENCODER_PATH):
            print("Run python train_model.py first")
            return False

        if not os.path.isfile(SCALER_PATH):
            print("Run python train_model.py first")
            return False
    except Exception as error:
        print(f"Error: startup file check failed: {error}")
        return False

    return True


def load_artifacts() -> Tuple[object, object, object]:
    # Load model, label encoder, and scaler for real-time prediction.
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
    except Exception as error:
        print(f"Error: unable to load model: {error}")
        sys.exit(1)

    try:
        print(f"Loading label encoder from {ENCODER_PATH}...")
        with open(ENCODER_PATH, mode="rb") as file_handle:
            encoder = pickle.load(file_handle)
    except Exception as error:
        print(f"Error: unable to load label encoder: {error}")
        sys.exit(1)

    try:
        print(f"Loading scaler from {SCALER_PATH}...")
        with open(SCALER_PATH, mode="rb") as file_handle:
            scaler = pickle.load(file_handle)
    except Exception as error:
        print(f"Error: unable to load scaler: {error}")
        sys.exit(1)

    return model, encoder, scaler


def create_detector() -> vision.HandLandmarker:
    # Create MediaPipe hand detector using the tasks API.
    try:
        base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.75,
            min_tracking_confidence=0.75,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        print("MediaPipe hand detector initialized.")
        return detector
    except Exception as error:
        print(f"Error: unable to initialize detector: {error}")
        sys.exit(1)


def normalize_landmarks(values: np.ndarray) -> np.ndarray:
    # Match training normalization: wrist-relative then hand-size scaling.
    points = values.reshape(21, 3).astype(np.float32)

    # Step 1: subtract wrist point (landmark 0) from all landmarks.
    wrist = points[0].copy()
    points[:, 0] = points[:, 0] - wrist[0]
    points[:, 1] = points[:, 1] - wrist[1]
    points[:, 2] = points[:, 2] - wrist[2]

    # Step 2: compute max pairwise distance as hand size.
    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    hand_size = float(np.max(distances))

    # Step 3: divide by hand size when valid.
    if hand_size > 1e-8:
        points = points / hand_size

    return points.reshape(-1)


def extract_features(hand_landmarks, scaler) -> np.ndarray:
    # Convert landmarks, normalize exactly as training, then apply scaler.
    features: List[float] = []
    for landmark in hand_landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])

    raw_values = np.array(features, dtype=np.float32)
    normalized_values = normalize_landmarks(raw_values)
    scaled_values = scaler.transform(normalized_values.reshape(1, -1))
    return scaled_values.astype(np.float32)


def predict_letter(model, encoder, features: np.ndarray) -> Tuple[str, float]:
    # Run classification and return predicted label and confidence.
    prediction = model.predict(features, verbose=0)[0]
    confidence = float(np.max(prediction))
    predicted_index = int(np.argmax(prediction))
    predicted_label = str(encoder.inverse_transform([predicted_index])[0])

    if confidence > CONFIDENCE_THRESHOLD:
        return predicted_label, confidence
    return "?", confidence


def draw_hand(frame: np.ndarray, hand_landmarks) -> Tuple[int, int, int, int]:
    # Draw landmark points, bones, and return a hand bounding box.
    frame_height, frame_width = frame.shape[:2]
    points = []

    for landmark in hand_landmarks:
        x_coord = int(landmark.x * frame_width)
        y_coord = int(landmark.y * frame_height)
        points.append((x_coord, y_coord))
        cv2.circle(frame, (x_coord, y_coord), 4, (0, 255, 0), -1)

    for start_index, end_index in HAND_CONNECTIONS:
        if start_index < len(points) and end_index < len(points):
            cv2.line(frame, points[start_index], points[end_index], (255, 200, 0), 2)

    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    if not x_values or not y_values:
        return 0, 0, 0, 0

    x_min = max(min(x_values) - 20, 0)
    y_min = max(min(y_values) - 20, 0)
    x_max = min(max(x_values) + 20, frame_width - 1)
    y_max = min(max(y_values) + 20, frame_height - 1)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return x_min, y_min, x_max, y_max


def apply_gesture(sentence: str, gesture: str) -> str:
    # Apply confirmed gesture edits to the sentence.
    if gesture == "space":
        return sentence + " "
    if gesture == "del":
        return sentence[:-1]
    if gesture == "nothing":
        return sentence
    return sentence + gesture


def draw_top_bar(frame: np.ndarray, detected_letter: str, confidence: float) -> None:
    # Render detected gesture and confidence in the top overlay.
    frame_width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (frame_width, 44), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Detected: {detected_letter}   Conf: {confidence * 100:.1f}%",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_progress_bar(frame: np.ndarray, progress_ratio: float) -> None:
    # Render hold-to-confirm progress directly below the top bar.
    frame_width = frame.shape[1]
    left = 20
    right = frame_width - 20
    top = 54
    bottom = 76

    cv2.rectangle(frame, (left, top), (right, bottom), (40, 40, 40), -1)
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 1)

    clamped = max(0.0, min(progress_ratio, 1.0))
    fill_width = int((right - left) * clamped)
    if fill_width > 0:
        cv2.rectangle(frame, (left, top), (left + fill_width, bottom), (0, 255, 0), -1)


def draw_bottom_overlay(frame: np.ndarray, sentence: str, fps_value: float) -> None:
    # Render sentence, controls, and FPS in bottom and top overlays.
    frame_height, frame_width = frame.shape[:2]

    bottom_top = frame_height - 60
    cv2.rectangle(frame, (0, bottom_top), (frame_width, frame_height), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Sentence: {sentence}",
        (12, frame_height - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Q = Quit   C = Clear",
        (12, frame_height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    fps_text = f"FPS: {fps_value:.1f}"
    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(
        frame,
        fps_text,
        (frame_width - text_size[0] - 16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    # Run startup checks and stop early if required files are missing.
    if not startup_checks():
        sys.exit(1)

    # Load model artifacts and initialize detector.
    model, encoder, scaler = load_artifacts()
    detector = create_detector()

    # Open webcam stream.
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Error: unable to open camera index {CAM_INDEX}")
        detector.close()
        sys.exit(1)

    print("Webcam started. Press Q to quit and C to clear sentence.")

    sentence = ""
    current_letter: Optional[str] = None
    letter_start_time = 0.0
    confirmed_this_hold = False

    last_detected_letter = "?"
    last_detected_confidence = 0.0

    frame_counter = 0
    previous_time = time.time()
    fps_value = 0.0

    try:
        while True:
            # Read frame and mirror it for intuitive interaction.
            success, frame = cap.read()
            if not success:
                print("Error: unable to read frame from webcam")
                break

            frame = cv2.flip(frame, 1)
            now = time.time()

            # Update FPS from real elapsed frame time.
            delta = now - previous_time
            if delta > 0:
                fps_value = 1.0 / delta
            previous_time = now

            # Convert frame for MediaPipe tasks API inference.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            try:
                result = detector.detect(mp_image)
            except Exception as error:
                print(f"Warning: detector failed on this frame: {error}")
                result = None

            progress_ratio = 0.0
            detected_letter = last_detected_letter
            detected_confidence = last_detected_confidence

            if result and result.hand_landmarks:
                # Draw landmarks and a green bounding box for the first detected hand.
                hand_landmarks = result.hand_landmarks[0]
                draw_hand(frame, hand_landmarks)

                # Run model prediction every second frame to reduce lag.
                frame_counter += 1
                if frame_counter % PREDICTION_STRIDE == 0:
                    try:
                        features = extract_features(hand_landmarks, scaler)
                        detected_letter, detected_confidence = predict_letter(model, encoder, features)
                        last_detected_letter = detected_letter
                        last_detected_confidence = detected_confidence
                    except Exception as error:
                        print(f"Warning: prediction failed on this frame: {error}")
                        detected_letter = "?"
                        detected_confidence = 0.0
                        last_detected_letter = "?"
                        last_detected_confidence = 0.0

                # Update hold-to-confirm state machine.
                if detected_letter != current_letter:
                    current_letter = detected_letter
                    letter_start_time = now
                    confirmed_this_hold = False
                elif letter_start_time == 0.0:
                    letter_start_time = now

                if current_letter not in {None, "", "?"} and not confirmed_this_hold:
                    elapsed = now - letter_start_time
                    progress_ratio = min(elapsed / HOLD_DURATION, 1.0)
                    if elapsed >= HOLD_DURATION:
                        sentence = apply_gesture(sentence, current_letter)
                        confirmed_this_hold = True
                        progress_ratio = 1.0
                elif confirmed_this_hold:
                    progress_ratio = 1.0
            else:
                # Reset hold state when no hand is visible.
                current_letter = None
                letter_start_time = 0.0
                confirmed_this_hold = False
                detected_letter = "?"
                detected_confidence = 0.0
                last_detected_letter = "?"
                last_detected_confidence = 0.0

            # Draw all overlays for status and controls.
            draw_top_bar(frame, detected_letter, detected_confidence)
            draw_progress_bar(frame, progress_ratio)
            draw_bottom_overlay(frame, sentence, fps_value)

            cv2.imshow("Sign Language to Text Converter", frame)

            # Process keyboard controls.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("Quit requested.")
                break
            if key in (ord("c"), ord("C")):
                sentence = ""
                current_letter = None
                letter_start_time = 0.0
                confirmed_this_hold = False
                print("Sentence cleared.")
    finally:
        # Release all resources cleanly.
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("Application closed cleanly.")


if __name__ == "__main__":
    main()
