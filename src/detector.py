from pathlib import Path
from typing import Any
import cv2
import numpy as np
from ultralytics import YOLO


class HazardDetector:
    def __init__(
        self,
        model_path: str | Path,
        device: str = "npu",
        imgsz: int = 640,
        conf: float = 0.30,
        iou: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.model = YOLO(str(self.model_path), task="segment")

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run YOLOv8-seg and return bounding boxes + exact polygon masks."""
        results = self.model.predict(
            frame,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            rect=False,  # Mandatory for fixed-shape Intel NPU OpenVINO graphs
            verbose=False,
        )[0]

        detections: list[dict[str, Any]] = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        polygons = results.masks.xy if results.masks is not None else [None] * len(results.boxes)

        for idx, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            conf_score = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = str(self.model.names.get(cls_id, "pothole"))

            poly = polygons[idx] if idx < len(polygons) else None
            if poly is not None and len(poly) >= 3:
                pts = poly.astype(np.float32)
                area_px = float(cv2.contourArea(pts))
                moments = cv2.moments(pts)
                y_c = float(moments["m01"] / moments["m00"]) if moments["m00"] > 0 else (y1 + y2) / 2.0
                poly_int = poly.astype(np.int32)
            else:
                area_px = max(0.0, (x2 - x1) * (y2 - y1))
                y_c = (y1 + y2) / 2.0
                poly_int = None

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf_score,
                    "label": label,
                    "area_px": area_px,
                    "y_c": y_c,
                    "polygon": poly_int,
                }
            )
        return detections