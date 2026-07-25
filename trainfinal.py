# Final Train Code
from pathlib import Path
from ultralytics import YOLO
import torch

# ---------- USER SETTINGS ----------
DATA_YAML  = r"C:\Users\Asus\Desktop\E-WasteGreyScale22.v6i.yolov8\data.yaml"

PROJECT    = "runs_ewaste"
RUN_NAME   = "yolov8s_ewastegrey22"   # change name to start a new run
MODEL_SIZE = "yolov8s.pt"       # use "yolov8n.pt" if VRAM is tight
EPOCHS     = 200
IMG_SIZE   = 240
SEED       = 0
WORKERS    = 0                  # 0 is safest on Windows; try 2 if you want faster loading
# -----------------------------------

DEVICE = 0 if torch.cuda.is_available() else "cpu"

def main():
    torch.manual_seed(SEED)

    last_ckpt = Path(PROJECT) / "detect" / RUN_NAME / "weights" / "last.pt"
    best_ckpt = Path(PROJECT) / "detect" / RUN_NAME / "weights" / "best.pt"

    try:
        if last_ckpt.exists():
            print(f"🔁 Resuming from {last_ckpt}")
            model = YOLO(str(last_ckpt))
            model.train(resume=True)
        else:
            print(f"🆕 Starting fresh ({MODEL_SIZE}) on device={DEVICE}")
            model = YOLO(MODEL_SIZE)
            model.train(
                data=DATA_YAML,
                epochs=EPOCHS,
                imgsz=IMG_SIZE,
                device=DEVICE,
                project=PROJECT,
                name=RUN_NAME,
                batch=-1,          # auto-batch to fit your 4 GB VRAM
                workers=WORKERS,   # Windows-safe
                seed=SEED,
                patience=50,
                save=True,
                save_period=1,
                pretrained=True,
                cos_lr=True,
                amp=True,          # mixed precision (good on RTX)
                deterministic=True,

                # Light aug to help rare classes
                mosaic=1.0,
                copy_paste=0.2,
                mixup=0.1,
                erasing=0.4,
                close_mosaic=10,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            )

        # Validate best model if present
        if best_ckpt.exists():
            print(f"✅ Best model: {best_ckpt}")
            best_model = YOLO(str(best_ckpt))
            best_model.val(data=DATA_YAML, device=DEVICE)

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user. You can rerun this script to resume from last.pt.")
    except RuntimeError as e:
        msg = str(e)
        if "CUDA out of memory" in msg:
            print("\n⚠️ CUDA OOM detected. Try one or more of these and rerun:")
            print("   - Set IMG_SIZE = 512 (or 448)")
            print("   - Use MODEL_SIZE = 'yolov8s.pt'")
            print("   - Set a fixed smaller batch, e.g., batch=8/6/4/2 (instead of -1)")
            print("   - Keep WORKERS = 0")
        raise

if __name__ == "__main__":
    main()
