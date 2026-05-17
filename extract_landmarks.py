import csv
import ctypes
import os
from typing import List, Optional, Tuple

import numpy as np

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

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Core extraction and output paths.
DATASET_PATH = os.path.join("dataset", "asl_alphabet_train", "asl_alphabet_train")
RAW_OUTPUT_CSV = "landmarks_data.csv"
NORMALIZED_OUTPUT_CSV = "landmarks_data_normalized.csv"
LANDMARKER_PATH = "hand_landmarker.task"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_header() -> List[str]:
    # Build standard CSV header with label and 63 features.
    header = ["label"]
    for index in range(21):
        header.extend([f"x{index}", f"y{index}", f"z{index}"])
    return header


def normalize_landmarks(values: np.ndarray) -> np.ndarray:
    # Normalize one 63-value sample to wrist-relative and hand-size scale.
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


def create_detector():
    # Build MediaPipe hand landmarker detector.
    if not os.path.isfile(LANDMARKER_PATH):
        raise FileNotFoundError(f"Hand landmarker model not found at {LANDMARKER_PATH}")

    base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks_from_image(image_path: str, detector) -> Optional[List[float]]:
    # Extract flat 63-value landmarks from one image.
    try:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        if not result.hand_landmarks:
            return None

        hand_landmarks = result.hand_landmarks[0]
        features: List[float] = []
        for landmark in hand_landmarks:
            features.extend([landmark.x, landmark.y, landmark.z])
        return features
    except Exception as error:
        print(f"Warning: failed to process image {image_path}: {error}")
        return None


def write_csv(path: str, header: List[str], rows: List[List[float]]) -> None:
    # Write a CSV file from prepared rows.
    with open(path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    # Validate dataset path before extraction.
    if not os.path.isdir(DATASET_PATH):
        print(f"Error: dataset folder not found at {DATASET_PATH}")
        return

    try:
        detector = create_detector()
    except Exception as error:
        print(f"Error: unable to create hand detector: {error}")
        return

    total_saved = 0
    total_skipped = 0
    header = build_header()
    raw_rows: List[List[float]] = []
    normalized_rows: List[List[float]] = []

    try:
        # Walk all label folders and collect raw + normalized rows.
        for label_folder in sorted(os.listdir(DATASET_PATH)):
            label_path = os.path.join(DATASET_PATH, label_folder)
            if not os.path.isdir(label_path):
                continue

            image_files = [
                file_name
                for file_name in sorted(os.listdir(label_path))
                if file_name.lower().endswith(SUPPORTED_EXTENSIONS)
            ]

            total_images = len(image_files)
            if total_images == 0:
                print(f"Skipping {label_folder}: no supported images found")
                continue

            for image_index, image_name in enumerate(image_files, start=1):
                image_path = os.path.join(label_path, image_name)
                landmarks = extract_landmarks_from_image(image_path, detector)

                if landmarks is None:
                    total_skipped += 1
                else:
                    values = np.array(landmarks, dtype=np.float32)
                    normalized_values = normalize_landmarks(values)

                    raw_rows.append([label_folder] + [float(value) for value in values])
                    normalized_rows.append([label_folder] + [float(value) for value in normalized_values])
                    total_saved += 1

                if image_index % 100 == 0 or image_index == total_images:
                    print(f"Processing {label_folder}: {image_index}/{total_images} done")

        # Save both raw and normalized CSV outputs.
        raw_path = os.path.join(RAW_OUTPUT_CSV)
        normalized_path = os.path.join(NORMALIZED_OUTPUT_CSV)
        write_csv(raw_path, header, raw_rows)
        write_csv(normalized_path, header, normalized_rows)
    except Exception as error:
        print(f"Error: extraction failed: {error}")
        return
    finally:
        try:
            detector.close()
        except Exception:
            pass

    print(f"Total saved: {total_saved}, Total skipped: {total_skipped}")
    print("Both CSV files saved successfully")


if __name__ == "__main__":
    main()
