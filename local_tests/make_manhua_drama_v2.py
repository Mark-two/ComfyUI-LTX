#!/usr/bin/env python3
"""Manhua drama assembler v2: all shots are videos (I2V + LivePortrait)."""
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("/home/kang/Documents/ComfyUI/output/local_manhua")
FPS = 24
W, H = 1080, 1920

# ── Shot definitions ──
SHOTS = [
    {"video": OUT_DIR / "manhua-shot1_00001.mp4", "text": "夜色天台，她独自伫立", "effect": "slow_zoom"},
    {"video": OUT_DIR / "liveportrait-test-manhua_closeup_00001_-d0-d0.45-e0.45-l0.45_00001.mp4", "text": "眼神坚定", "effect": "none"},
    {"video": OUT_DIR / "manhua-shot2_00001.mp4", "text": "每一步都是力量", "effect": "speed_lines"},
    {"video": OUT_DIR / "manhua-shot3_00001.mp4", "text": "转身，等待命运", "effect": "none"},
    {"video": None, "text": "AI 漫剧", "effect": "title_card", "bg": OUT_DIR / "heroine-hq_00001_.png"},
]


def smooth_video(src, dst, target_fps=32):
    """Frame interpolation via ffmpeg minterpolate."""
    if dst.exists():
        return dst
    subprocess.run([
        "ffmpeg", "-y", "-i", str(src),
        "-filter:v", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dst),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst


def resize_cover(img, w, h):
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    resized = cv2.resize(img, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_LANCZOS4)
    y = (resized.shape[0] - h) // 2
    x = (resized.shape[1] - w) // 2
    return resized[y:y + h, x:x + w]


def add_subtitle(frame, text):
    if not text:
        return frame
    h, w = frame.shape[:2]
    y = int(h * 0.85)
    over = frame.copy()
    cv2.rectangle(over, (0, y - 55), (w, y + 25), (0, 0, 0), -1)
    frame = cv2.addWeighted(over, 0.4, frame, 0.6, 0)
    scale = 1.3
    th = 3
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        cv2.putText(frame, text, (60 + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (60, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 245, 230), th, cv2.LINE_AA)
    return frame


def add_speed_lines(frame, strength):
    h, w = frame.shape[:2]
    lines = np.zeros_like(frame, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    for i in range(40):
        angle = i / 40 * np.pi * 2
        r1, r2 = 40, int(400 + 300 * strength)
        x0, y0 = int(cx + np.cos(angle) * r1), int(cy + np.sin(angle) * r1)
        x1, y1 = int(cx + np.cos(angle) * r2), int(cy + np.sin(angle) * r2)
        cv2.line(lines, (x0, y0), (x1, y1), (255, 255, 255), 1, cv2.LINE_AA)
    lines = cv2.GaussianBlur(lines, (0, 0), 2)
    return cv2.addWeighted(frame, 1.0, lines, 0.3 * strength, 0)


def render_video_shot(video_path, text, effect, duration=0):
    """Read video frame-by-frame, apply effects, return frames."""
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    raw = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        raw.append(f)
    cap.release()
    if not raw:
        return frames

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 16
    if duration <= 0:
        duration = len(raw) / src_fps
    total = int(duration * FPS)

    for i in range(total):
        progress = i / max(1, total - 1)
        src_idx = int(progress * (len(raw) - 1))
        frame = resize_cover(raw[src_idx], W, H)
        if effect == "slow_zoom":
            scale = 0.9 + 0.1 * progress
            m = cv2.getRotationMatrix2D((W / 2, H / 2), 0, scale)
            frame = cv2.warpAffine(frame, m, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        elif effect == "speed_lines":
            frame = add_speed_lines(frame, 0.4 + 0.6 * np.sin(progress * np.pi))
            if i % 4 == 0:
                sh = np.random.randint(-2, 3, 2)
                m = np.float32([[1, 0, sh[0]], [0, 1, sh[1]]])
                frame = cv2.warpAffine(frame, m, (W, H), borderMode=cv2.BORDER_REPLICATE)
        frame = add_subtitle(frame, text)
        frames.append(frame)
    return frames


def render_title_card(bg_path, text, duration=2.5):
    frames = []
    bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
    bg = resize_cover(bg, W, H)
    bg = cv2.GaussianBlur(bg, (0, 0), 15)
    total = int(duration * FPS)
    for i in range(total):
        progress = i / max(1, total - 1)
        frame = bg.copy()
        # Fade in text
        alpha = min(1.0, progress * 2)
        scale = 2.5
        th = 5
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, th)[0]
        x, y = (W - size[0]) // 2, H // 2
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3)]:
            cv2.putText(frame, text, (x + dx, y + dy), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 220, 180), th, cv2.LINE_AA)
        if alpha < 1.0:
            frame = cv2.addWeighted(bg, 1.0 - alpha, frame, alpha, 0)
        frames.append(frame)
    return frames


def main():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    final = OUT_DIR / f"manhua-drama-v2-{stamp}.mp4"
    writer = cv2.VideoWriter(str(final), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    for i, shot in enumerate(SHOTS):
        print(f"Shot {i+1}: {shot['text']}")

        if shot["effect"] == "title_card":
            frames = render_title_card(shot["bg"], shot["text"])
        else:
            # Smooth the I2V video first
            src = Path(shot["video"])
            smooth_path = src.parent / f"{src.stem}_smooth.mp4"
            smooth_video(src, smooth_path, target_fps=32)
            frames = render_video_shot(smooth_path, shot["text"], shot["effect"], duration=2.5 if "shot" in str(src) else 4.0)

        for f in frames:
            writer.write(f)
        print(f"  → {len(frames)} frames ({len(frames)/FPS:.1f}s)")

    writer.release()
    print(f"\nDone: {final}")


if __name__ == "__main__":
    main()
