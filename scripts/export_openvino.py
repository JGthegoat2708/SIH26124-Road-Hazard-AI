from pathlib import Path
import argparse
import shutil
import time
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent


def benchmark_model(model: YOLO, device: str, imgsz: int, runs: int = 30) -> tuple[float, float]:
    dummy_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    # Warmup pass (compiles OpenVINO graph on NPU / initializes CUDA kernels)
    model.predict(dummy_frame, device=device, imgsz=imgsz, rect=False, verbose=False)

    t0 = time.perf_counter()
    for _ in range(runs):
        model.predict(dummy_frame, device=device, imgsz=imgsz, rect=False, verbose=False)
    avg_ms = ((time.perf_counter() - t0) / runs) * 1000.0
    return avg_ms, 1000.0 / avg_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8-seg to OpenVINO and benchmark devices.")
    parser.add_argument("--weights", type=str, default="models/best.pt")
    parser.add_argument("--precision", type=str, choices=["fp16", "int8"], default="fp16")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--data", type=str, default=None, help="Required only for int8 calibration")
    args = parser.parse_args()

    weights_path = (REPO_ROOT / args.weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}. Run scripts/download_assets.py first.")

    # Clean up old mismatched folder name if it exists
    old_dir = weights_path.parent / "best_openvino_fp16"
    if old_dir.exists():
        shutil.rmtree(old_dir)

    print(f"[INFO] Loading {weights_path} ...")
    pt_model = YOLO(str(weights_path))

    if args.precision == "fp16":
        print("[INFO] Exporting static 640x640 OpenVINO FP16 IR...")
        exported_dir = Path(
            pt_model.export(
                format="openvino",
                quantize=16,
                imgsz=args.imgsz,
                batch=1,
                dynamic=False,
            )
        )
        # MUST end in '_openvino_model' so Ultralytics detects the OpenVINO runtime
        target_dir = weights_path.parent / "best_fp16_openvino_model"
    else:
        if not args.data:
            raise ValueError("--data path/to/data.yaml is required for INT8 NPU export.")
        print("[INFO] Exporting static 640x640 NPU-symmetric OpenVINO INT8 IR...")
        exported_dir = Path(
            pt_model.export(
                format="openvino",
                quantize=8,
                device="npu",
                data=args.data,
                imgsz=args.imgsz,
                batch=1,
                dynamic=False,
            )
        )
        target_dir = weights_path.parent / "best_int8_npu_openvino_model"

    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(exported_dir), str(target_dir))
    print(f"[SUCCESS] OpenVINO IR saved to: {target_dir}\n")

    print("--- Running 30-Frame Latency Benchmark (640x640, batch=1) ---")
    if torch.cuda.is_available():
        gpu_ms, gpu_fps = benchmark_model(pt_model, "cuda:0", args.imgsz)
        print(f"  RTX 5060 Ti (PyTorch cuda:0) : {gpu_ms:6.2f} ms/frame | {gpu_fps:6.1f} FPS")

    ov_model = YOLO(str(target_dir), task="segment")
    for dev in ["npu", "xpu", "cpu"]:
        try:
            ms, fps = benchmark_model(ov_model, dev, args.imgsz)
            print(f"  OpenVINO ({dev:<6})            : {ms:6.2f} ms/frame | {fps:6.1f} FPS")
        except Exception as exc:
            print(f"  OpenVINO ({dev:<6})            : FAILED -> {exc}")


if __name__ == "__main__":
    main()