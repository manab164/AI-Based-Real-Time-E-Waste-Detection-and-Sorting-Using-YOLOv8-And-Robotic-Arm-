import cv2, time, os, serial
from collections import Counter, deque
from pathlib import Path
from ultralytics import YOLO
import torch
import sys
import atexit

# ===================== USER SETTINGS =====================
MODEL_WEIGHTS = r"C:\Users\Asus\Desktop\Minor Project\runs_ewaste\yolov8s_ewastegrey22\weights\best.pt"
CONF_THRESH   = 0.30
IMG_SIZE      = 640
SAVE_DIR      = r"C:\Users\Asus\Desktop\Minor Project\hazard_snaps"
CAM_INDEX     = 0
ARDUINO_PORT  = "COM5"   # <<< change to your actual COM port
ARDUINO_BAUD  = 9600
CLASS_PRIORITY = ["high", "moderate", "low"]
REQUIRED_STABLE_FRAMES = 5
MAX_WAIT_ARM_SECONDS   = 20
# =========================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

ser = None  # so we can access from atexit

def send_stop():
    """Try to send emergency STOP 'X' to Arduino on exit."""
    global ser
    try:
        if ser and ser.is_open:
            ser.write(b'X\n')
            ser.flush()
            time.sleep(0.05)
    except Exception:
        pass

atexit.register(send_stop)

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

def open_serial(port, baud):
    global ser
    ser = serial.Serial(port, baud, timeout=0.1)
    time.sleep(2.0)  # Arduino reset
    while ser.in_waiting:
        ser.readline()
    return ser

def wait_for_arm_done(timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        if ser and ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if line.startswith("Done:"):
                return True
        time.sleep(0.02)
    return False

def send_bin_command(hazard):
    m = {'high':'H','moderate':'M','low':'L'}
    cmd = m.get(hazard.lower())
    if not cmd: return False
    ser.write((cmd + "\n").encode())
    return True

def banner(img, text, org, scale=0.9, color=(255,255,255), thick=2, bg=True):
    if bg:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = org
        cv2.rectangle(img, (x-6, y-h-6), (x+w+6, y+6), (0,0,0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def main():
    # Model
    wp = Path(MODEL_WEIGHTS)
    if not wp.exists():
        raise FileNotFoundError(f"Model not found: {wp}")
    model = YOLO(str(wp)).to(DEVICE)
    ids, _ = find_label_ids(model.names)
    print("Hazard IDs:", ids)

    # Camera
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    # Serial
    open_serial(ARDUINO_PORT, ARDUINO_BAUD)
    print("Serial open on", ARDUINO_PORT)

    cv2.namedWindow("E-waste Pick & Place", cv2.WINDOW_NORMAL)
    state = "DETECT"
    stable = deque(maxlen=REQUIRED_STABLE_FRAMES)
    t0, frames = time.time(), 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            frames += 1

            if state == "DETECT":
                r = model(frame, conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False, device=DEVICE)[0]
                counts = Counter()
                if r.boxes is not None and len(r.boxes):
                    for c in r.boxes.cls:
                        cid = int(c.item())
                        if ids["high"] is not None and cid == ids["high"]: counts["high"] += 1
                        elif ids["moderate"] is not None and cid == ids["moderate"]: counts["moderate"] += 1
                        elif ids["low"] is not None and cid == ids["low"]: counts["low"] += 1

                # priority target
                target = next((c for c in CLASS_PRIORITY if counts[c] > 0), None)

                # stabilize
                stable.append(target)
                stable_ok = len(stable)==REQUIRED_STABLE_FRAMES and all(x==target and x is not None for x in stable)

                annotated = r.plot()
                hi, mo, lo = counts["high"], counts["moderate"], counts["low"]
                color = (255,255,255)
                if hi>0: color=(0,0,255)
                elif mo>0: color=(0,255,255)
                elif lo>0: color=(0,200,0)

                fps = frames / max(1e-6, (time.time()-t0))
                banner(annotated, f"High:{hi}  Moderate:{mo}  Low:{lo}", (18,40), color=color)
                banner(annotated, f"{fps:.1f} FPS", (18,75), scale=0.7, color=(200,200,200), bg=False)
                if target:
                    banner(annotated, f"Target: {target.upper()} ({len(stable)}/{REQUIRED_STABLE_FRAMES})",
                           (18,110), scale=0.8, color=(180,220,255))
                cv2.imshow("E-waste Pick & Place", annotated)

                if stable_ok:
                    if send_bin_command(target):
                        snap = Path(SAVE_DIR)/f"{target}_{int(time.time()*1000)}.jpg"
                        cv2.imwrite(str(snap), annotated)
                        stable.clear()
                        state = "ARM_BUSY"
                    else:
                        stable.clear()

            else:  # ARM_BUSY
                done = wait_for_arm_done(timeout=MAX_WAIT_ARM_SECONDS)
                print("Arm cycle:", "complete" if done else "timeout")
                state = "DETECT"
                time.sleep(0.2)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                print("Quitting... sending STOP")
                send_stop()
                break

    except KeyboardInterrupt:
        print("KeyboardInterrupt — sending STOP")

    finally:
        # always send emergency stop on exit
        send_stop()
        try:
            if ser and ser.is_open:
                ser.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
