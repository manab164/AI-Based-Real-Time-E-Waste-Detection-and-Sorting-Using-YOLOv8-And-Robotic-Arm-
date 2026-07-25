from ultralytics import YOLO
import torch, cv2, os, time
from collections import Counter
from pathlib import Path

# ===================== USER SETTINGS =====================
MODEL_WEIGHTS = r"C:\Users\Asus\Desktop\Minor Project\runs_ewaste\yolov8s_ewastegrey22\weights\best.pt"
CONF_THRESH   = 0.30    
IMG_SIZE      = 640
SAVE_DIR      = r"C:\Users\Asus\Desktop\Minor Project\hazard_snaps"
AUTO_SNAPSHOT = True     # save a frame whenever a High hazard is present
BEEP_ON_HIGH  = True     # play a short beep on High hazard (Windows)
SHOW_CLASS_COUNTS = True # show per-class counts (without confidences)
MAX_COUNT_LINES   = 12   # max lines to print for class counts (top-N by count)
CAM_INDEX     = 0        # default webcam
# =========================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# Optional Windows beep
def beep():
    try:
        import winsound
        winsound.Beep(1500, 200)  # freq, duration(ms)
    except Exception:
        pass

def find_label_ids(names_dict):
    # model.names can be list or dict
    if isinstance(names_dict, list):
        id2name = {i: str(n).lower() for i, n in enumerate(names_dict)}
    else:
        id2name = {int(k): str(v).lower() for k, v in names_dict.items()}

    name2id = {v: k for k, v in id2name.items()}

    def get_id(target):
        t = target.lower()
        if t in name2id:
            return name2id[t]
        for i, n in id2name.items():
            if t in n:
                return i
        return None

    ids = {
        "high": get_id("hh"),
        "moderate": get_id("mh"),
        "low": get_id("lh"),
    }
    # Also return the original (human) names for display
    human_names = {}
    if isinstance(names_dict, list):
        human_names = {i: str(n) for i, n in enumerate(names_dict)}
    else:
        human_names = {int(k): str(v) for k, v in names_dict.items()}
    return ids, human_names

def put_text(img, text, org, scale=0.9, color=(255,255,255), thick=2, bg=True):
    if bg:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = org
        cv2.rectangle(img, (x-6, y-h-6), (x+w+6, y+6), (0,0,0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def main():
    # ---- Load model ----
    weights_path = Path(MODEL_WEIGHTS)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    model.to(DEVICE)

    ids, human_names = find_label_ids(model.names)
    print(f"Loaded names: {human_names}")
    print(f"Hazard class IDs: {ids}")

    # ---- Open camera ----
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Try CAM_INDEX=1 or close other apps using the camera.")

    window = "E-waste Hazard Monitor"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Stream predictions from the camera
    frame_gen = model.predict(
        source=CAM_INDEX,
        stream=True,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        device=DEVICE,
        verbose=False,
        show=False,
        vid_stride=1,
        show_labels=True,
        show_conf=False,   # << hide confidence on boxes
        show_boxes=True
    )

    t0 = time.time()
    n = 0
    for result in frame_gen:
        n += 1

        # Draw boxes/labels WITHOUT confidences
        try:
            annotated = result.plot(conf=False, labels=True, boxes=True)  # works on newer Ultralytics
        except TypeError:
            annotated = result.plot()  # fallback; show_conf=False above already hides conf

        # Count hazard levels and all classes on this frame
        counts_hazard = Counter()
        counts_all = Counter()
        if result.boxes is not None and len(result.boxes) > 0:
            for cls_tensor in result.boxes.cls:
                cls_id = int(cls_tensor.item())
                counts_all[cls_id] += 1
                if ids.get("high") is not None and cls_id == ids["high"]:
                    counts_hazard["High"] += 1
                elif ids.get("moderate") is not None and cls_id == ids["moderate"]:
                    counts_hazard["Moderate"] += 1
                elif ids.get("low") is not None and cls_id == ids["low"]:
                    counts_hazard["Low"] += 1

        hi, mo, lo = counts_hazard["High"], counts_hazard["Moderate"], counts_hazard["Low"]
        status = f"High: {hi}   Moderate: {mo}   Low: {lo}"

        # Color coding
        if hi > 0:
            color = (0, 0, 255)      # red
            if BEEP_ON_HIGH:
                beep()
        elif mo > 0:
            color = (0, 255, 255)    # yellow
        elif lo > 0:
            color = (0, 200, 0)      # green
        else:
            color = (255, 255, 255)  # white

        put_text(annotated, status, (18, 40), scale=0.9, color=color, thick=2)

        # FPS banner (no conf text)
        fps = n / max(1e-6, (time.time() - t0))
        put_text(annotated, f"{fps:.1f} FPS", (18, 75), scale=0.7, color=(200,200,200), thick=2, bg=False)

        # Optional: per-class counts (top-right)
        if SHOW_CLASS_COUNTS and len(counts_all) > 0:
            # sort by count desc, then class name
            items = sorted(counts_all.items(), key=lambda kv: (-kv[1], human_names.get(kv[0], str(kv[0]))))
            items = items[:MAX_COUNT_LINES]
            x0 = annotated.shape[1] - 320  # right area
            y0 = 40
            put_text(annotated, "Counts:", (x0, y0), scale=0.8, color=(255,255,255), thick=2)
            y = y0 + 28
            for cls_id, cnt in items:
                label = human_names.get(cls_id, str(cls_id))
                put_text(annotated, f"{label}: {cnt}", (x0, y), scale=0.8, color=(180,220,255), thick=1, bg=False)
                y += 24

        # Auto snapshot on high hazard
        if AUTO_SNAPSHOT and hi > 0:
            snap_path = Path(SAVE_DIR) / f"high_{int(time.time()*1000)}.jpg"
            cv2.imwrite(str(snap_path), annotated)

        cv2.imshow(window, annotated)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            snap_path = Path(SAVE_DIR) / f"snap_{int(time.time()*1000)}.jpg"
            cv2.imwrite(str(snap_path), annotated)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
