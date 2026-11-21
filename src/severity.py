from typing import Any


def compute_severity(
    area_px: float,
    y_c: float,
    frame_w: int,
    frame_h: int,
    minor_max: float = 1.2,
    moderate_max: float = 3.0,
) -> dict[str, Any]:
    frame_area = max(1, frame_w * frame_h)
    raw_area_pct = (area_px / frame_area) * 100.0

    # Linear perspective boost: horizon (0.50 -> 1.25x) vs bumper (0.95 -> 1.025x)
    y_norm = max(0.0, min(1.0, y_c / max(1, frame_h)))
    perspective_weight = 1.5 - 0.5 * y_norm
    score = raw_area_pct * perspective_weight

    if score < minor_max:
        tier = "MINOR"
    elif score < moderate_max:
        tier = "MODERATE"
    else:
        tier = "CRITICAL"

    return {
        "mask_area_pct": round(raw_area_pct, 2),
        "perspective_weight": round(perspective_weight, 3),
        "severity_score": round(score, 2),
        "severity_tier": tier,
    }