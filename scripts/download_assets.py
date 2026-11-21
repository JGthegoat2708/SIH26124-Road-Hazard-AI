from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    target_pt = MODELS_DIR / "best.pt"
    if target_pt.exists():
        print(f"[OK] Starter model already exists at: {target_pt}")
        return

    print("[INFO] Downloading starter YOLOv8n pothole segmentation weights from Hugging Face...")
    cached_path = hf_hub_download(
        repo_id="keremberke/yolov8n-pothole-segmentation",
        filename="best.pt",
    )
    shutil.copy2(cached_path, target_pt)
    size_mb = target_pt.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Saved starter weights to {target_pt} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()