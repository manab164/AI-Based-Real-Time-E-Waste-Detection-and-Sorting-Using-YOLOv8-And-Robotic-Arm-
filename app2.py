# app2_freeze.py
import os
import time
import atexit
import threading
from collections import Counter, deque
from pathlib import Path

from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import torch
import logging

try:
    import serial
except Exception:
    serial = None

# ----------------- USER SETTINGS (edit these) -----------------
MODEL_PATH = r"C:\Users\Asus\Desktop\Minor Project\runs_ewaste\yolov8s_ewastegrey22\weights\best.pt"
CONF_THRESH = 0.30
IMG_SIZE = 640
CAM_INDEX = 0
SAVE_DIR = r"C:\Users\Asus\Desktop\Minor Project\hazard_snaps"
ARDUINO_PORT = "COM5"
ARDUINO_BAUD = 9600
CLASS_PRIORITY = ["high", "moderate", "low"]
REQUIRED_STABLE_FRAMES = 5
MAX_WAIT_ARM_SECONDS = 20
# ----------------------------------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Shared globals
frame_lock = threading.Lock()
annotated_frame = None            # latest live annotated numpy frame
latest_detections = {"High": 0, "Moderate": 0, "Low": 0}
_device = 0 if torch.cuda.is_available() else "cpu"

# Freeze control for the stream
is_frozen = False
frozen_frame_bytes = None         # holds JPEG bytes to serve while frozen

ser = None

def send_stop():
    global ser
    try:
        if ser and getattr(ser, "is_open", False):
            ser.write(b'X\n')
            ser.flush()
            time.sleep(0.05)
    except Exception:
        pass

atexit.register(send_stop)

# Serial helpers
def open_serial(port, baud):
    global ser
    if serial is None:
        logging.warning("pyserial not installed — serial functionality disabled.")
        return None
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2.0)
        while ser.in_waiting:
            ser.readline()
        logging.info(f"Serial opened on {port} @ {baud}")
    except Exception as e:
        logging.warning(f"Could not open serial port {port}: {e}")
        ser = None
    return ser

def wait_for_arm_done(timeout=20):
    if not ser:
        return False
    start = time.time()
    while time.time() - start < timeout:
        try:
            if ser.in_waiting:
                line = ser.readline().decode(errors="ignore").strip()
                if line.startswith("Done:"):
                    return True
        except Exception:
            pass
        time.sleep(0.02)
    return False

def send_bin_command(hazard):
    if not ser:
        logging.debug("Serial not available — not sending command.")
        return False
    mapping = {'high': 'H', 'moderate': 'M', 'low': 'L'}
    cmd = mapping.get(hazard.lower())
    if not cmd:
        return False
    try:
        ser.write((cmd + "\n").encode())
        logging.info(f"Sent command '{cmd}' for hazard '{hazard}'")
        return True
    except Exception as e:
        logging.warning(f"Failed to send command to serial: {e}")
        return False

# label id finder
def find_label_ids(names_dict):
    if isinstance(names_dict, list):
        id2name = {i: str(n).lower() for i, n in enumerate(names_dict)}
    else:
        id2name = {int(k): str(v).lower() for k, v in names_dict.items()}
    def get_id(target):
        t = target.lower()
        for i, n in id2name.items():
            if t == n or t in n:
                return i
        return None
    ids = {"high": get_id("hh"), "moderate": get_id("mh"), "low": get_id("lh")}
    return ids, {i: str(v) for i, v in id2name.items()}

# Flask app
app = Flask(__name__)

# Load model
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
logging.info("Loading YOLO model...")
model = YOLO(MODEL_PATH).to(_device)
label_ids, id2name = find_label_ids(model.names)
logging.info(f"Model loaded. Hazard label ids: {label_ids}")

# Camera
def open_cam(index=CAM_INDEX, w=1280, h=720):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap

cap = open_cam()
if not cap.isOpened():
    logging.error("Camera failed to open. Check CAM_INDEX or permissions.")
else:
    logging.info("Camera opened successfully.")

# Detection thread
def detection_loop():
    global annotated_frame, latest_detections, cap, is_frozen, frozen_frame_bytes
    state = "DETECT"
    stable = deque(maxlen=REQUIRED_STABLE_FRAMES)
    frames = 0
    t0 = time.time()

    while True:
        if cap is None or not cap.isOpened():
            cap = open_cam()
            time.sleep(0.5)
            continue

        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        frames += 1

        if state == "DETECT":
            r = model(frame, conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False, device=_device)[0]

            counts = Counter()
            if r.boxes is not None and len(r.boxes):
                for c in r.boxes.cls:
                    cid = int(c.item())
                    if label_ids["high"] is not None and cid == label_ids["high"]:
                        counts["high"] += 1
                    elif label_ids["moderate"] is not None and cid == label_ids["moderate"]:
                        counts["moderate"] += 1
                    elif label_ids["low"] is not None and cid == label_ids["low"]:
                        counts["low"] += 1

            target = next((c for c in CLASS_PRIORITY if counts[c] > 0), None)

            # stabilize
            stable.append(target)
            stable_ok = (len(stable) == REQUIRED_STABLE_FRAMES and
                         all(x == target and x is not None for x in stable))

            # annotated frame
            annotated = r.plot()
            hi, mo, lo = counts["high"], counts["moderate"], counts["low"]

            # overlay
            fps = frames / max(1e-6, (time.time() - t0))
            cv2.putText(annotated, f"High:{hi}  Moderate:{mo}  Low:{lo}", (18,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(annotated, f"{fps:.1f} FPS", (18,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            if target:
                cv2.putText(annotated, f"Target: {target.upper()} ({len(stable)}/{REQUIRED_STABLE_FRAMES})",
                            (18,110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,220,255), 2)

            # update latest detections
            latest_detections = {"High": hi, "Moderate": mo, "Low": lo}

            # update annotated frame unless we are frozen (freeze only during ARM_BUSY)
            with frame_lock:
                if not is_frozen:
                    annotated_frame = annotated.copy()

            if stable_ok:
                # encode the current annotated frame to JPEG and set frozen bytes immediately
                _, buf = cv2.imencode('.jpg', annotated)
                with frame_lock:
                    frozen_frame_bytes = buf.tobytes()
                    is_frozen = True

                # send command to Arduino
                sent = send_bin_command(target)

                # save snapshot always
                snap = Path(SAVE_DIR)/f"{target}_{int(time.time()*1000)}.jpg"
                try:
                    cv2.imwrite(str(snap), annotated)
                except Exception as e:
                    logging.warning(f"Failed to save snapshot: {e}")

                stable.clear()
                if sent:
                    state = "ARM_BUSY"
                else:
                    # if serial not available, hold frozen frame for a small time then unfreeze
                    logging.info("Serial not available — holding freeze short then resume.")
                    time.sleep(1.0)
                    with frame_lock:
                        is_frozen = False
                        frozen_frame_bytes = None
                    state = "DETECT"

        else:  # ARM_BUSY
            done = wait_for_arm_done(timeout=MAX_WAIT_ARM_SECONDS)
            logging.info("Arm cycle: %s", "complete" if done else "timeout")
            # unfreeze stream and resume detection
            with frame_lock:
                is_frozen = False
                frozen_frame_bytes = None
            state = "DETECT"
            time.sleep(0.2)

# Start services
def start_services():
    open_serial(ARDUINO_PORT, ARDUINO_BAUD)
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    logging.info("Detection thread started.")

# Stream generator uses frozen_frame_bytes when is_frozen is True
import numpy as np

def generate_frames_for_stream():
    global annotated_frame, is_frozen, frozen_frame_bytes
    while True:
        with frame_lock:
            if is_frozen and frozen_frame_bytes is not None:
                frame_bytes = frozen_frame_bytes
            else:
                img = annotated_frame.copy() if annotated_frame is not None else None
                if img is None:
                    placeholder = 255 * (np.zeros((360,640,3), dtype='uint8'))
                    _, buffer = cv2.imencode('.jpg', placeholder)
                    frame_bytes = buffer.tobytes()
                else:
                    _, buffer = cv2.imencode('.jpg', img)
                    frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

# Routes
@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_for_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def detections():
    return jsonify({
        "High": {"count": latest_detections.get("High", 0), "bin": "Bin 1"},
        "Moderate": {"count": latest_detections.get("Moderate", 0), "bin": "Bin 2"},
        "Low": {"count": latest_detections.get("Low", 0), "bin": "Bin 3"},
    })

if __name__ == "__main__":
    start_services()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
