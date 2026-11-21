from datetime import datetime, timezone
import math
from typing import Any


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * earth_r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


class GeoDeduplicator:
    def __init__(
        self,
        start_lat: float = 26.88510,
        start_lon: float = 75.79050,
        lat_step: float = 0.000003,
        lon_step: float = 0.000002,
        dedup_radius_m: float = 5.0,
        corridor: str = "Tonk Road (Rambagh - Tonk Phatak)",
    ) -> None:
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.lat_step = lat_step
        self.lon_step = lon_step
        self.dedup_radius_m = dedup_radius_m
        self.corridor = corridor
        self.records: list[dict[str, Any]] = []
        self._date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")

    def gps_for_frame(self, frame_idx: int) -> tuple[float, float]:
        lat = round(self.start_lat + frame_idx * self.lat_step, 6)
        lon = round(self.start_lon + frame_idx * self.lon_step, 6)
        return lat, lon

    def upsert_track(self, candidate: dict[str, Any]) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for rec in self.records:
            dist = haversine_meters(
                candidate["latitude"], candidate["longitude"], rec["latitude"], rec["longitude"]
            )
            if dist <= self.dedup_radius_m:
                rec["sighting_count"] += 1
                rec["last_seen_utc"] = now_iso
                if candidate["severity_score"] > rec["severity_score"]:
                    rec["severity_score"] = candidate["severity_score"]
                    rec["severity_tier"] = candidate["severity_tier"]
                    rec["mask_area_pct"] = candidate["mask_area_pct"]
                if candidate["confidence"] > rec["confidence"]:
                    rec["confidence"] = candidate["confidence"]
                    rec["crop_path"] = candidate["crop_path"]
                return rec

        hazard_id = f"JPR-{self._date_tag}-{len(self.records) + 1:03d}"
        new_rec = {
            "hazard_id": hazard_id,
            "track_id": candidate["track_id"],
            "timestamp_utc": now_iso,
            "last_seen_utc": now_iso,
            "latitude": candidate["latitude"],
            "longitude": candidate["longitude"],
            "corridor": self.corridor,
            "severity_tier": candidate["severity_tier"],
            "severity_score": candidate["severity_score"],
            "mask_area_pct": candidate["mask_area_pct"],
            "confidence": candidate["confidence"],
            "sighting_count": 1,
            "crop_path": candidate["crop_path"],
            "inference_device": candidate["inference_device"],
        }
        self.records.append(new_rec)
        return new_rec