from pathlib import Path
import shutil
import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "dataset"
MODELS_DIR = REPO_ROOT / "models"
ASSETS_DIR = REPO_ROOT / "assets"


def prepare_dataset() -> Path:
    """Locate (or download) the 1,986-image pothole segmentation dataset and build a clean data.yaml."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if dataset is already unzipped in data/dataset/
    search_root = DATA_DIR
    if not any(DATA_DIR.rglob("*.jpg")) and not any(DATA_DIR.rglob("*.png")):
        print("[INFO] data/dataset/ is empty. Downloading dataset via kagglehub...")
        import kagglehub

        downloaded_path = Path(
            kagglehub.dataset_download("raunakkesharwani/pothole-image-segmentation-dataset")
        )
        print(f"[OK] Downloaded to cache: {downloaded_path}")
        search_root = downloaded_path

    # Auto-detect train and val/valid image directories
    train_imgs = None
    val_imgs = None
    for candidate in search_root.rglob("images"):
        if not candidate.is_dir():
            continue
        parent_name = candidate.parent.name.lower()
        if parent_name == "train":
            train_imgs = candidate.resolve()
        elif parent_name in {"val", "valid", "validation"}:
            val_imgs = candidate.resolve()

    if train_imgs is None or val_imgs is None:
        raise FileNotFoundError(
            f"Could not find train/images and valid/images inside {search_root}. "
            "Please unzip the Kaggle dataset into data/dataset/."
        )

    yaml_path = DATA_DIR / "data_local.yaml"
    data_cfg = {
        "train": str(train_imgs),
        "val": str(val_imgs),
        "names": {0: "pothole"},
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)

    print(f"[OK] Generated clean YOLO config: {yaml_path}")
    print(f"     Train images: {train_imgs}")
    print(f"     Val images  : {val_imgs}")
    return yaml_path


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected! Make sure your .venv with PyTorch cu128 is active.")

    print(f"[GPU] Training on: {torch.cuda.get_device_name(0)}")
    yaml_path = prepare_dataset()

    # Start from COCO-pretrained YOLOv8n-seg base weights
    model = YOLO("yolov8n-seg.pt")

    results = model.train(
        data=str(yaml_path),
        epochs=40,
        imgsz=640,
        batch=32,          # Sweet spot for RTX 5060 Ti 16GB VRAM
        device=0,
        workers=4,
        patience=12,       # Stops early if accuracy plateaus
        project=str(REPO_ROOT / "runs" / "segment"),
        name="sih26124_pothole_seg",
        exist_ok=True,
    )

    run_dir = Path(results.save_dir)
    trained_best = run_dir / "weights" / "best.pt"
    if not trained_best.exists():
        raise FileNotFoundError(f"Expected trained weights at {trained_best}")

    # Backup Day-1 starter model before overwriting models/best.pt
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    current_best = MODELS_DIR / "best.pt"
    starter_backup = MODELS_DIR / "starter_keremberke_best.pt"
    if current_best.exists() and not starter_backup.exists():
        shutil.copy2(current_best, starter_backup)
        print(f"[BACKUP] Saved Day-1 starter weights to {starter_backup}")

    shutil.copy2(trained_best, current_best)
    print(f"[SUCCESS] Upgraded models/best.pt with your custom-trained weights!")

    # Copy proof charts into assets/ for your GitHub README and PPT
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    proof_files = [
        "results.png",
        "confusion_matrix.png",
        "MaskPR_curve.png",
        "BoxPR_curve.png",
        "val_batch0_pred.jpg",
    ]
    for fname in proof_files:
        src_file = run_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, ASSETS_DIR / fname)
            print(f"[ASSET] Copied {fname} -> assets/{fname}")


if __name__ == "__main__":
    main()
