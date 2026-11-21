from typing import Any
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    inter_w = max(0.0, x_right - x_left)
    inter_h = max(0.0, y_bottom - y_top)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


class HazardTracker:
    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
    ) -> None:
        use_cuda = torch.cuda.is_available()
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder="mobilenet",
            embedder_gpu=use_cuda,
            half=use_cuda,
        )

    def update(
        self, detections: list[dict[str, Any]], frame: np.ndarray
    ) -> list[dict[str, Any]]:
        raw_inputs = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            raw_inputs.append(([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)], det["conf"], det["label"]))

        tracks = self.tracker.update_tracks(raw_inputs, frame=frame)
        active_tracks: list[dict[str, Any]] = []

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            ltrb = [float(v) for v in track.to_ltrb()]
            best_det = None
            best_iou = 0.15
            for det in detections:
                iou = _bbox_iou(ltrb, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            active_tracks.append(
                {
                    "track_id": int(track.track_id),
                    "bbox": ltrb,
                    "label": (best_det["label"] if best_det else (track.get_det_class() or "pothole")),
                    "conf": (best_det["conf"] if best_det else float(track.get_det_conf() or 0.5)),
                    "area_px": (
                        best_det["area_px"]
                        if best_det
                        else max(0.0, (ltrb[2] - ltrb[0]) * (ltrb[3] - ltrb[1]))
                    ),
                    "y_c": (best_det["y_c"] if best_det else (ltrb[1] + ltrb[3]) / 2.0),
                    "polygon": (best_det["polygon"] if best_det else None),
                }
            )
        return active_tracks