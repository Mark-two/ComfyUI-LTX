#!/usr/bin/env python3
import math
import time
from pathlib import Path

import cv2
import numpy as np


FACE_VIDEO = Path("/home/kang/Documents/ComfyUI/output/local_manhua/liveportrait-test-20260607_201952_01-d0-d0.45-e0.45-l0.45_00001.mp4")
OUT_DIR = Path("/home/kang/Documents/ComfyUI/output/local_manhua")


def glow_overlay(frame: np.ndarray, t: float) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame, dtype=np.float32)
    colors = [
        (255, 70, 170),
        (80, 180, 255),
        (180, 90, 255),
    ]
    for idx, color in enumerate(colors):
        cx = int(w * (0.2 + 0.6 * ((math.sin(t * 1.7 + idx * 2.0) + 1) / 2)))
        cy = int(h * (0.10 + 0.10 * math.sin(t * 2.1 + idx)))
        cv2.circle(overlay, (cx, cy), int(w * 0.55), color, -1, cv2.LINE_AA)
    blur = cv2.GaussianBlur(overlay, (0, 0), 70)
    return cv2.addWeighted(frame.astype(np.float32), 1.0, blur, 0.16, 0).clip(0, 255).astype(np.uint8)


def add_spotlights(frame: np.ndarray, t: float) -> np.ndarray:
    h, w = frame.shape[:2]
    layer = frame.copy()
    for idx, x0 in enumerate((int(w * 0.16), int(w * 0.84))):
        swing = int(math.sin(t * 1.8 + idx * math.pi) * w * 0.22)
        top = (x0, 0)
        bottom = (int(w * 0.5 + swing), h)
        pts = np.array([top, (bottom[0] - int(w * 0.16), h), (bottom[0] + int(w * 0.16), h)], np.int32)
        color = (120, 170, 255) if idx == 0 else (255, 110, 190)
        cv2.fillConvexPoly(layer, pts, color)
    return cv2.addWeighted(layer, 0.18, frame, 0.82, 0)


def add_subtitle(frame: np.ndarray, text: str) -> np.ndarray:
    h, w = frame.shape[:2]
    band_h = 58
    y0 = h - band_h - 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (36, y0), (w - 36, y0 + band_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.42, frame, 0.58, 0)
    cv2.putText(frame, text, (58, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 245, 235), 2, cv2.LINE_AA)
    return frame


def main() -> None:
    cap = cv2.VideoCapture(str(FACE_VIDEO))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {FACE_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit("no frames")

    h, w = frames[0].shape[:2]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    silent = OUT_DIR / f"stage-sing-test-silent-{stamp}.mp4"
    final = OUT_DIR / f"stage-sing-test-{stamp}.mp4"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    lyrics = [
        "夜色点燃心跳",
        "我站在光里歌唱",
        "命运随节拍翻涌",
        "这一刻由我登场",
    ]
    total = len(frames)
    for i, frame in enumerate(frames):
        t = i / fps
        beat = math.sin(t * math.tau * 1.4)
        scale = 1.0 + 0.018 * beat
        angle = 1.0 * math.sin(t * math.tau * 0.7)
        tx = 6 * math.sin(t * math.tau * 0.9)
        ty = 4 * math.sin(t * math.tau * 1.2 + 1.1)
        m = cv2.getRotationMatrix2D((w / 2, h * 0.48), angle, scale)
        m[0, 2] += tx
        m[1, 2] += ty
        frame = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        frame = glow_overlay(frame, t)
        frame = add_spotlights(frame, t)
        lyric = lyrics[min(len(lyrics) - 1, int(i / max(1, total / len(lyrics))))]
        frame = add_subtitle(frame, lyric)
        writer.write(frame)
    writer.release()

    # Keep audio simple and local: synthetic stage beat, not a real song.
    import subprocess
    duration = total / fps
    cmd = [
        "ffmpeg", "-y",
        "-i", str(silent),
        "-f", "lavfi", "-t", f"{duration:.3f}",
        "-i", "sine=frequency=110:sample_rate=44100",
        "-filter_complex", "[1:a]volume=0.08,atrim=0:999[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-shortest", str(final),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(final)


if __name__ == "__main__":
    main()
