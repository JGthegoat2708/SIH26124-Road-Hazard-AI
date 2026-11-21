from pathlib import Path
import subprocess
import imageio_ffmpeg


def transcode_to_h264(raw_mp4: Path, web_mp4: Path) -> None:
    """Convert OpenCV mp4v output to browser-compatible H.264 + yuv420p."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(raw_mp4),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-preset",
        "fast",
        "-crf",
        "23",
        str(web_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)