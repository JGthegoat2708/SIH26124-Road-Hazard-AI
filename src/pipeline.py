from pathlib import Path
import argparse
import json
import time
import cv2
import numpy as np
import yaml

from src.detector import HazardDetector
from src.geo_dedup import GeoDeduplicator
from src.severity import compute_severity
from src.tracker import HazardTracker
from src.video_utils import transcode_to_h264

REPO_ROOT = Path(__file__).resolve().parent.parent

TIER_COLORS = {
    "CRITICAL": (0, 0, 255),     # Red (BGR)
    "MODERATE": (0, 165, 255),   # Orange
    "MINOR": (0, 220, 80),       # Green
}


def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    model_path: str,
    source_path: str,
    device: str,
    config_path: str = "configs/config.yaml",
    show_window: bool = False,
) -> Path:
    cfg = load_config(REPO_ROOT / config_path)
    outputs_dir = REPO_ROOT / "outputs"
    crops_dir = outputs_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Clear old crops from previous runs
    for old_crop in crops_dir.glob("track_*.jpg"):
        old_crop.unlink(missing_ok=True)

    video_in = Path(source_path)
    if not video_in.is_absolute():
        video_in = (REPO_ROOT / video_in).resolve()
    if not video_in.exists():
        raise FileNotFoundError(f"Video not found: {video_in}")

    cap = cv2.VideoCapture(str(video_in))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_mp4 = outputs_dir / "_raw_annotated.mp4"
    web_mp4 = outputs_dir / "annotated_web.mp4"
    writer = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

    detector = HazardDetector(
        model_path=REPO_ROOT / model_path if not Path(model_path).is_absolute() else Path(model_path),
        device=device,
        imgsz=cfg["model"]["imgsz"],
        conf=cfg["model"]["conf_threshold"],
        iou=cfg["model"]["iou_threshold"],
    )
    tracker = HazardTracker(
        max_age=cfg["tracker"]["max_age"],
        n_init=cfg["tracker"]["n_init"],
        max_iou_distance=cfg["tracker"]["max_iou_distance"],
    )
    dedup = GeoDeduplicator(
        start_lat=cfg["geo"]["start_lat"],
        start_lon=cfg["geo"]["start_lon"],
        lat_step=cfg["geo"]["lat_step_per_frame"],
        lon_step=cfg["geo"]["lon_step_per_frame"],
        dedup_radius_m=cfg["geo"]["dedup_radius_meters"],
        corridor=cfg["geo"]["default_corridor"],
    )

    # Store best frame snapshot per DeepSORT track_id (maximizing confidence * area)
    track_best: dict[int, dict] = {}
    frame_idx = 0
    t_pipeline_start = time.perf_counter()

    print(f"[PIPELINE] Source: {video_in.name} ({width}x{height} @ {fps_in:.1f} FPS, {total_frames} frames)")
    print(f"[PIPELINE] Model: {model_path} | Device: {device}")

    while cap.isOpened():
        t_frame = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        lat, lon = dedup.gps_for_frame(frame_idx)

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        overlay = frame.copy()
        has_masks = False

        for trk in tracks:
            tid = trk["track_id"]
            x1, y1, x2, y2 = [int(v) for v in trk["bbox"]]
            sev = compute_severity(
                area_px=trk["area_px"],
                y_c=trk["y_c"],
                frame_w=width,
                frame_h=height,
                minor_max=cfg["severity"]["minor_max"],
                moderate_max=cfg["severity"]["moderate_max"],
            )
            color = TIER_COLORS[sev["severity_tier"]]

            if trk["polygon"] is not None:
                cv2.fillPoly(overlay, [trk["polygon"]], color)
                has_masks = True

            # Keep sharpest evidence snapshot per unique track_id
            quality = trk["conf"] * max(1.0, trk["area_px"])
            prev = track_best.get(tid)
            if prev is None or quality > prev["quality"]:
                pad = 12
                cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                cx2, cy2 = min(width, x2 + pad), min(height, y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]
                crop_rel = f"outputs/crops/track_{tid:03d}.jpg"
                if crop.size > 0:
                    cv2.imwrite(str(REPO_ROOT / crop_rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])

                track_best[tid] = {
                    "quality": quality,
                    "track_id": tid,
                    "latitude": lat,
                    "longitude": lon,
                    "severity_tier": sev["severity_tier"],
                    "severity_score": sev["severity_score"],
                    "mask_area_pct": sev["mask_area_pct"],
                    "confidence": round(trk["conf"], 2),
                    "crop_path": crop_rel,
                    "inference_device": device,
                }

            # Draw tracking box and severity badge
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f"ID #{tid} | {sev['severity_tier']} ({sev['severity_score']:.2f})"
            cv2.putText(frame, tag, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if has_masks:
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Draw top Edge AI Telemetry HUD
        inst_fps = 1.0 / max(0.001, time.perf_counter() - t_frame)
        cv2.rectangle(frame, (0, 0), (width, 42), (18, 18, 18), -1)
        hud = (
            f"SIH26124 EDGE AI | DEV: {device.upper()} | {inst_fps:4.1f} FPS | "
            f"GPS: {lat:.5f}N, {lon:.5f}E | TRACKED: {len(track_best)}"
        )
        cv2.putText(frame, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 180), 2)

        writer.write(frame)
        if show_window:
            cv2.imshow("SIH26124 Edge Pipeline (Press Q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{total_frames or '?'} frames | Unique tracks: {len(track_best)}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Spatial 5m Haversine deduplication across finalized tracks
    for tid in sorted(track_best.keys()):
        dedup.upsert_track(track_best[tid])

    hazards_json = outputs_dir / "hazards.json"
    with open(hazards_json, "w", encoding="utf-8") as f:
        json.dump(dedup.records, f, indent=2)

    print("[PIPELINE] Transcoding annotated video to H.264 for Streamlit browser playback...")
    transcode_to_h264(raw_mp4, web_mp4)
    raw_mp4.unlink(missing_ok=True)

    elapsed = time.perf_counter() - t_pipeline_start
    avg_fps = frame_idx / max(0.001, elapsed)
    print(f"[SUCCESS] Done in {elapsed:.1f}s ({avg_fps:.1f} pipeline FPS)")
    print(f"          Hazards logged : {len(dedup.records)} -> {hazards_json}")
    print(f"          Web video saved: {web_mp4}")
    return hazards_json


def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26124 Edge AI Road Hazard Pipeline")
    parser.add_argument("--model", type=str, default="models/best_fp16_openvino_model")
    parser.add_argument("--source", type=str, default="data/videos/sample_jaipur.mp4")
    parser.add_argument("--device", type=str, default="npu", help="npu, xpu, cpu, or cuda:0")
    parser.add_argument("--show", action="store_true", help="Display live OpenCV preview window")
    args = parser.parse_args()

    run_pipeline(
        model_path=args.model,
        source_path=args.source,
        device=args.device,
        show_window=args.show,
    )


if __name__ == "__main__":
    main()