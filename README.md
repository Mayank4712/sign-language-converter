# 🤟 Sign Language to Text Converter
A real-time American Sign Language (ASL) alphabet converter that uses MediaPipe hand landmarks and a neural network to turn webcam gestures into text.

## Features
- ✅ Real-time webcam gesture detection
- ✅ MediaPipe hand landmark extraction
- ✅ Hold-to-confirm input to reduce accidental predictions
- ✅ Automatic text-to-speech with `pyttsx3`
- ✅ Green hand bounding boxes and live confidence display
- ✅ Trainable pipeline with saved model and label encoder
- ✅ Accuracy and loss curves saved after training

## Tech Stack
| Component | Purpose |
|---|---|
| Python 3.10+ | Core language |
| OpenCV | Webcam capture and UI overlays |
| MediaPipe | Hand landmark detection |
| TensorFlow / Keras | Neural network training and inference |
| NumPy | Numeric arrays and feature vectors |
| Pandas | CSV data loading and cleanup |
| scikit-learn | Label encoding, train/test split, metrics |
| pyttsx3 | Offline text-to-speech |
| matplotlib | Training curve plots |

## Installation
1. Clone or open this project in VS Code.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Kaggle ASL Alphabet dataset and paste it into the `dataset/` folder:
   https://www.kaggle.com/datasets/grassknoted/asl-alphabet
4. Extract landmarks:
   ```bash
   python extract_landmarks.py
   ```
5. Train the model:
   ```bash
   python train_model.py
   ```
6. Run the app:
   ```bash
   python app.py
   ```

## Folder Structure
```text
sign-language-converter/
├── dataset/
├── model/
│   ├── sign_model.h5
│   └── label_encoder.pkl
├── extract_landmarks.py
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

## How Hold-to-Confirm Works
The webcam shows the currently detected gesture, but it does not add text immediately. A gesture must stay stable for `HOLD_DURATION` seconds before it is accepted.

- `space` adds a space
- `del` removes the last character
- `nothing` does nothing
- Any letter from `A` to `Z` is appended to the sentence
- Low-confidence predictions display `?` and are ignored until the hand becomes clearer

## Keyboard Shortcuts
| Key | Action |
|---|---|
| Q | Quit the application |
| C | Clear the current sentence |


## Common Errors and Fixes
### 1. `Model not found`
- Make sure `python train_model.py` completed successfully.
- Confirm these files exist in `model/`: `sign_model.h5` and `label_encoder.pkl`.

### 2. Webcam not opening
- Close apps that are already using the camera.
- Change `CAM_INDEX` in `app.py` from `0` to `1` or `2` if your webcam is on a different device index.

### 3. MediaPipe hand not detected
- Improve lighting and keep your hand fully inside the frame.
- Make sure the dataset folder is organized as class folders like `dataset/A`, `dataset/B`, and so on.

## Future Improvements
- Add word-level prediction with sequence modeling
- Support dynamic signing beyond alphabet gestures
- Improve robustness under poor lighting and motion blur
- Add a custom calibration mode for different camera setups
- Export the model to TensorFlow Lite for faster edge deployment

## Why This Project Matters
Sign language tools make communication more accessible for deaf and hard-of-hearing users, and they also help non-signers learn and interact more respectfully. A lightweight webcam-based converter can be useful in classrooms, assistive technology demos, and accessibility-focused prototypes.
