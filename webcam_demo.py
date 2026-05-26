"""
Live webcam demo for the art classifier.
Hold a painting / sculpture / drawing inside the ROI box.

Controls:
    Q       -> Quit
    S       -> Save a screenshot to static/uploads/
    SPACE   -> Pause / resume prediction
"""

import os
import time
import pickle
import numpy as np
import cv2
import tensorflow as tf

# ---------- Config ----------
MODEL_PATH = 'model/classifier.keras'
META_PATH = 'model/metrics.pkl'
SCREENSHOT_DIR = 'static/uploads'
ROI_SIZE = 360               # square ROI in pixels on the live frame
SMOOTHING = 5                # average last N predictions to reduce flicker
CONF_THRESHOLD = 0.55        # below this, show "Uncertain - may not be art"

# ---------- Load model + metadata ----------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(META_PATH, 'rb') as f:
    metadata = pickle.load(f)

CLASS_NAMES = metadata['class_names']
IMG_SIZE = tuple(metadata['img_size'])
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

print(f"Classes: {CLASS_NAMES}")
print(f"Input size: {IMG_SIZE}")

# ---------- Open webcam ----------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow backend = faster on Windows
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check if another app is using it.")

# Optional: request a sensible resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

paused = False
recent_preds = []
fps_t0 = time.time()
fps_frames = 0
fps_value = 0.0

# Colour palette per class (BGR)
PALETTE = [
    (255, 160, 0),   # drawings    - blue/orange
    (0, 200, 255),   # engraving   - yellow
    (180, 80, 220),  # iconography - purple
    (60, 200, 60),   # painting    - green
    (60, 60, 220),   # sculpture   - red
]

print("\nWebcam ready. Press Q to quit, S for screenshot, SPACE to pause.\n")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame")
        break

    frame = cv2.flip(frame, 1)  # mirror so it feels natural
    h, w = frame.shape[:2]

    # ---------- ROI box in the centre ----------
    x1 = (w - ROI_SIZE) // 2
    y1 = (h - ROI_SIZE) // 2
    x2 = x1 + ROI_SIZE
    y2 = y1 + ROI_SIZE
    roi = frame[y1:y2, x1:x2]

    # ---------- Predict ----------
    if not paused:
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE)
        batch = np.expand_dims(resized / 255.0, axis=0).astype(np.float32)
        preds = model.predict(batch, verbose=0)[0]

        recent_preds.append(preds)
        if len(recent_preds) > SMOOTHING:
            recent_preds.pop(0)

    smoothed = np.mean(recent_preds, axis=0) if recent_preds else np.zeros(len(CLASS_NAMES))
    top_idx = int(np.argmax(smoothed))
    top_conf = float(smoothed[top_idx])
    top_label = CLASS_NAMES[top_idx] if top_conf >= CONF_THRESHOLD else "uncertain"

    # ---------- Draw ROI box ----------
    is_uncertain = top_conf < CONF_THRESHOLD
    box_colour = (60, 200, 255) if is_uncertain else PALETTE[top_idx]  # yellow when uncertain
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 3)
    cv2.putText(frame, "Place artwork inside the box",
                (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ---------- Prediction header ----------
    header_h = 80 if is_uncertain else 50
    cv2.rectangle(frame, (0, 0), (w, header_h), (0, 0, 0), -1)
    header = f"{top_label.upper()}   {top_conf*100:.1f}%"
    cv2.putText(frame, header, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, box_colour, 2)
    if is_uncertain:
        cv2.putText(frame, "may not be one of the 5 trained art categories",
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # ---------- Probability bars (bottom-left panel) ----------
    panel_w = 280
    panel_h = 30 * len(CLASS_NAMES) + 20
    panel_x = 10
    panel_y = h - panel_h - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    for i, cls in enumerate(CLASS_NAMES):
        prob = float(smoothed[i])
        y = panel_y + 25 + i * 30
        bar_w = int(prob * (panel_w - 130))
        colour = PALETTE[i]
        cv2.putText(frame, f"{cls[:11]:<11}", (panel_x + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.rectangle(frame, (panel_x + 110, y - 12),
                      (panel_x + 110 + bar_w, y + 2), colour, -1)
        cv2.putText(frame, f"{prob*100:4.1f}%", (panel_x + panel_w - 60, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

    # ---------- FPS + status ----------
    fps_frames += 1
    if time.time() - fps_t0 >= 1.0:
        fps_value = fps_frames / (time.time() - fps_t0)
        fps_t0 = time.time()
        fps_frames = 0
    status = "PAUSED" if paused else f"{fps_value:.1f} FPS"
    cv2.putText(frame, status, (w - 130, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("Art Classifier - Live Webcam Demo", frame)

    # ---------- Keys ----------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif key == ord('s'):
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SCREENSHOT_DIR, f"webcam_{ts}.png")
        cv2.imwrite(path, frame)
        print(f"Saved screenshot -> {path}")

cap.release()
cv2.destroyAllWindows()
print("Closed.")
