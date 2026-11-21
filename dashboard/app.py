import base64
import json
import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Ensure repo root is importable when launched via `streamlit run dashboard/app.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUTS_DIR = REPO_ROOT / "outputs"
LIVE_JSON = OUTPUTS_DIR / "hazards.json"
SAMPLE_JSON = OUTPUTS_DIR / "sample_hazards.json"
WEB_VIDEO = OUTPUTS_DIR / "annotated_web.mp4"

TIER_COLORS = {
    "CRITICAL": "#ef4444",   # Red
    "MODERATE": "#f97316",   # Orange
    "MINOR": "#22c55e",      # Green
}

st.set_page_config(
    page_title="SIH26124 | Jaipur ICCC Road Hazard Command",
    page_icon="🚌",
    layout="wide",
)


def load_hazards() -> tuple[list[dict], str]:
    if LIVE_JSON.exists():
        with open(LIVE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data:
            return data, "Live Edge AI Output (outputs/hazards.json)"
    if SAMPLE_JSON.exists():
        with open(SAMPLE_JSON, "r", encoding="utf-8") as f:
            return json.load(f), "Sample Fallback Data (outputs/sample_hazards.json)"
    return [], "No Data Found"


def encode_crop_b64(crop_rel_path: str) -> str | None:
    if not crop_rel_path:
        return None
    full_path = REPO_ROOT / crop_rel_path
    if not full_path.exists():
        return None
    return base64.b64encode(full_path.read_bytes()).decode("utf-8")


# --- Header ---
st.title("🚌 SIH26124: Edge AI Road Hazard Command Center")
st.caption("Jaipur Smart City ICCC Prototype • YOLOv8-Seg + DeepSORT + 5m Haversine Spatial Deduplication")

records, data_source_label = load_hazards()

# --- Sidebar: Filters & Live Edge Inference Runner ---
with st.sidebar:
    st.header("🎛️ Municipal Filters")
    st.info(f"**Source:** {data_source_label}")

    all_tiers = ["CRITICAL", "MODERATE", "MINOR"]
    selected_tiers = st.multiselect("Severity Tiers", all_tiers, default=all_tiers)

    corridors = sorted({r.get("corridor", "Unknown") for r in records}) if records else []
    selected_corridors = st.multiselect("Jaipur Corridors", corridors, default=corridors)

    st.divider()
    st.header("⚡ Run Live Edge Inference")
    uploaded_video = st.file_uploader("Upload Road Dashcam Clip (.mp4)", type=["mp4"])
    device_choice = st.selectbox(
        "Target Accelerator",
        options=["npu", "xpu", "cuda:0", "cpu"],
        index=0,
        help="npu/xpu/cpu run OpenVINO FP16 IR; cuda:0 runs PyTorch on your RTX 5060 Ti.",
    )

    if st.button("Process Video Now", type="primary", disabled=uploaded_video is None):
        from src.pipeline import run_pipeline

        upload_path = REPO_ROOT / "data" / "videos" / "_uploaded_demo.mp4"
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        upload_path.write_bytes(uploaded_video.read())

        model_to_use = (
            "models/best.pt"
            if device_choice.startswith("cuda")
            else "models/best_fp16_openvino_model"
        )
        if not (REPO_ROOT / model_to_use).exists():
            st.error(f"Model missing: `{model_to_use}`. Run Mission 2 export scripts first.")
        else:
            with st.spinner(f"Running YOLOv8-Seg + DeepSORT on {device_choice.upper()}..."):
                run_pipeline(
                    model_path=model_to_use,
                    source_path=str(upload_path),
                    device=device_choice,
                    show_window=False,
                )
            st.success("Inference complete! Refreshing command map...")
            st.rerun()

# --- Filter Records ---
filtered = [
    r
    for r in records
    if r.get("severity_tier") in selected_tiers
    and (not selected_corridors or r.get("corridor") in selected_corridors)
]

# --- Zone 1: Top KPI Strip ---
total_hazards = len(filtered)
critical_cnt = sum(1 for r in filtered if r.get("severity_tier") == "CRITICAL")
moderate_cnt = sum(1 for r in filtered if r.get("severity_tier") == "MODERATE")
active_device = (
    filtered[0].get("inference_device", "N/A").upper() if filtered else "N/A"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Unique Verified Hazards", total_hazards)
k2.metric("🔴 Critical Priority", critical_cnt)
k3.metric("🟠 Moderate Priority", moderate_cnt)
k4.metric("Active Edge Device", active_device)
k5.metric("Cellular Data Saved", "99.97%", help="Modeled 39.91 GiB vs 117.9 TiB raw fleet video/month")

st.divider()

# --- Zone 2 & 3: Interactive Jaipur Map + H.264 Annotated Video ---
map_col, video_col = st.columns([1.25, 1.0])

with map_col:
    st.subheader("🗺️ Jaipur Geo-Tagged Hazard Map (5m Haversine Dedup)")
    if filtered:
        center_lat = sum(r["latitude"] for r in filtered) / len(filtered)
        center_lon = sum(r["longitude"] for r in filtered) / len(filtered)
    else:
        center_lat, center_lon = 26.88510, 75.79050

    # Rebuild map cleanly per filter state (never cache folium.Map in session_state)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="OpenStreetMap")

    for rec in filtered:
        tier = rec.get("severity_tier", "MINOR")
        color = TIER_COLORS.get(tier, "#3b82f6")
        b64_img = encode_crop_b64(rec.get("crop_path", ""))
        img_html = (
            f'<br><img src="data:image/jpeg;base64,{b64_img}" width="210" '
            f'style="border-radius:6px;margin-top:6px;border:1px solid #ccc;" />'
            if b64_img
            else "<br><i>(No crop thumbnail available)</i>"
        )

        popup_html = (
            f"<div style='font-family:sans-serif;font-size:12px;width:225px;'>"
            f"<b>{rec['hazard_id']}</b> "
            f"<span style='color:{color};font-weight:bold;'>[{tier}]</span><br>"
            f"<b>Score:</b> {rec['severity_score']} ({rec['mask_area_pct']}% area)<br>"
            f"<b>Confidence:</b> {rec['confidence']} | <b>Passes:</b> {rec['sighting_count']}<br>"
            f"<b>GPS:</b> {rec['latitude']:.5f}, {rec['longitude']:.5f}"
            f"{img_html}"
            f"</div>"
        )

        folium.CircleMarker(
            location=[rec["latitude"], rec["longitude"]],
            radius=9 if tier == "CRITICAL" else (7 if tier == "MODERATE" else 6),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{rec['hazard_id']} ({tier} | Score {rec['severity_score']})",
        ).add_to(m)

    # returned_objects=[] prevents Streamlit rerun loops when panning or zooming
    st_folium(m, height=460, use_container_width=True, returned_objects=[])

with video_col:
    st.subheader("🎥 Edge AI Annotated Dashcam Feed")
    if WEB_VIDEO.exists():
        st.video(str(WEB_VIDEO))
        st.caption("H.264 (`libx264` + `yuv420p`) browser-ready output from `src/pipeline.py`.")
    else:
        st.warning("No rendered video yet. Run `src/pipeline.py` or upload a clip in the sidebar.")

# --- Zone 4: Cropped Evidence Gallery ---
st.divider()
st.subheader("📸 Peak-Frame Pothole Evidence Crops")

crops_available = [
    r for r in filtered if r.get("crop_path") and (REPO_ROOT / r["crop_path"]).exists()
]
if crops_available:
    # Sort highest severity first
    crops_sorted = sorted(crops_available, key=lambda x: x["severity_score"], reverse=True)
    cols = st.columns(4)
    for idx, rec in enumerate(crops_sorted[:8]):
        with cols[idx % 4]:
            st.image(str(REPO_ROOT / rec["crop_path"]), use_container_width=True)
            st.markdown(
                f"**{rec['hazard_id']}** • `{rec['severity_tier']}`  \n"
                f"Score: `{rec['severity_score']}` | Conf: `{rec['confidence']}` | Passes: `{rec['sighting_count']}`"
            )
else:
    st.info("No cropped evidence images found for the current filter selection.")

# --- Zone 5: Municipal Work-Order Table & CSV Export ---
st.divider()
st.subheader("📋 Jaipur Municipal Work-Order Queue")

if filtered:
    df = pd.DataFrame(filtered)
    display_cols = [
        "hazard_id",
        "severity_tier",
        "severity_score",
        "mask_area_pct",
        "confidence",
        "sighting_count",
        "corridor",
        "latitude",
        "longitude",
        "inference_device",
        "timestamp_utc",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    df_view = df[existing_cols].sort_values(by="severity_score", ascending=False)
    st.dataframe(df_view, use_container_width=True, hide_index=True)

    csv_bytes = df_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Export Municipal Work-Order CSV",
        data=csv_bytes,
        file_name="jaipur_iccc_road_hazards.csv",
        mime="text/csv",
    )
else:
    st.warning("No hazards match the selected filters.")