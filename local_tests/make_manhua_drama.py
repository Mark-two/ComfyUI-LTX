#!/usr/bin/env python3
"""Motion comic / manhua-style short drama assembler.
Combines still images, LivePortrait face animation, Ken Burns effects,
subtitles, and edge-tts voiceover into a single video.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("/home/kang/Documents/ComfyUI/output/local_manhua")

# --- Shot definitions ---
SHOTS = [
    {
        "image": str(OUT_DIR / "heroine-hq_00001_.png"),
        "type": "still",
        "duration": 3.0,
        "effect": "ken_burns",  # slow zoom in
        "text": "夜色下的城市天台",
        "voice": "夜色下的城市天台，她独自伫立，",
    },
    {
        "image": str(OUT_DIR / "liveportrait-test-manhua_closeup_00001_-d0-d0.45-e0.45-l0.45_00001.mp4"),
        "type": "video",
        "duration": None,  # use video duration
        "effect": "none",
        "text": "",
        "voice": "眼神坚定，等待着命运的降临。",
    },
    {
        "image": str(OUT_DIR / "heroine-hq_00002_.png"),
        "type": "still",
        "duration": 2.5,
        "effect": "speed_lines",  # action lines + shake
        "text": "",
        "voice": "每一步，都是向前的力量。",
    },
    {
        "image": str(OUT_DIR / "heroine-hq_00001_.png"),
        "type": "still",
        "duration": 2.0,
        "effect": "fade_out",
        "text": "漫剧 · AI 制作",
        "voice": "",
    },
]

FPS = 24
WIDTH, HEIGHT = 1080, 1920  # vertical 9:16 short video format
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_SIZE = 48


def resize_cover(img, w, h):
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    resized = cv2.resize(img, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_LANCZOS4)
    y = (resized.shape[0] - h) // 2
    x = (resized.shape[1] - w) // 2
    return resized[y:y + h, x:x + w]


def add_subtitle(frame, text, y_ratio=0.88):
    if not text:
        return frame
    h, w = frame.shape[:2]
    y = int(h * y_ratio)
    # Shadow band
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y - 60), (w, y + 30), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    font_scale = FONT_SIZE / 30.0
    thickness = max(2, int(FONT_SIZE / 20))
    # Text outline (black)
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        cv2.putText(frame, text, (50 + dx, y + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (50, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 240, 220), thickness, cv2.LINE_AA)
    return frame


def add_title(frame, text):
    if not text:
        return frame
    h, w = frame.shape[:2]
    scale = 1.8
    thickness = 4
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)[0]
    x = (w - size[0]) // 2
    y = h // 2
    for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3)]:
        cv2.putText(frame, text, (x + dx, y + dy),
                    cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_DUPLEX, scale, (255, 220, 180), thickness, cv2.LINE_AA)
    return frame


def add_ken_burns(frame, progress):
    """Slow zoom in from 85% to 100% over the shot duration."""
    scale = 0.85 + 0.15 * progress
    h, w = frame.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    return cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def add_speed_lines(frame, strength):
    """Add anime-style speed lines for action shots."""
    h, w = frame.shape[:2]
    lines = np.zeros_like(frame, dtype=np.uint8)
    center = (w // 2, h // 2)
    for i in range(30):
        angle = i / 30 * 3.14159 * 2
        r1 = 50
        r2 = int(500 + 300 * strength)
        dx = int(np.cos(angle) * (r2 - r1))
        dy = int(np.sin(angle) * (r2 - r1))
        x0 = int(center[0] + np.cos(angle) * r1)
        y0 = int(center[1] + np.sin(angle) * r1)
        cv2.line(lines, (x0, y0), (x0 + dx, y0 + dy), (255, 255, 255), 2, cv2.LINE_AA)
    lines = cv2.GaussianBlur(lines, (0, 0), 1.5)
    return cv2.addWeighted(frame, 1.0, lines, 0.25 * strength, 0)


def render_shot(shot, fps, width, height):
    """Render a single shot to a list of frames."""
    frames = []
    path = Path(shot["image"])
    
    if shot["type"] == "video" and path.suffix == ".mp4":
        cap = cv2.VideoCapture(str(path))
        raw_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_frames.append(frame)
        cap.release()
        duration = shot["duration"] or (len(raw_frames) / 16.0)  # LivePortrait outputs at 16fps
        total_frames = int(duration * fps)
        for i in range(total_frames):
            src_idx = int(i / total_frames * len(raw_frames))
            if src_idx >= len(raw_frames):
                src_idx = len(raw_frames) - 1
            frame = resize_cover(raw_frames[src_idx], width, height)
            progress = i / total_frames
            frames.append(frame)
    else:
        # Still image
        src = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if src is None:
            print(f"WARNING: cannot read {path}")
            return frames
        duration = shot["duration"]
        total_frames = int(duration * fps)
        for i in range(total_frames):
            progress = i / max(1, total_frames - 1)
            frame = resize_cover(src, width, height)
            effect = shot.get("effect", "none")
            if effect == "ken_burns":
                frame = add_ken_burns(frame, progress)
            elif effect == "speed_lines":
                frame = add_speed_lines(frame, 0.5 + 0.5 * np.sin(progress * np.pi))
                # slight camera shake
                if i % 3 == 0:
                    shake = np.random.randint(-3, 4, 2)
                    m = np.float32([[1, 0, shake[0]], [0, 1, shake[1]]])
                    frame = cv2.warpAffine(frame, m, (width, height), borderMode=cv2.BORDER_REPLICATE)
            elif effect == "fade_out":
                alpha = 1.0 - progress * 0.5
                frame = cv2.addWeighted(frame, alpha, np.zeros_like(frame), 1 - alpha, 0)
            frames.append(frame)

    # Add text overlays
    text = shot.get("text", "")
    for i, frame in enumerate(frames):
        mid = len(frames) // 2
        if text and i > mid - len(frames) // 3:
            frame = add_subtitle(frame, text)
        elif text == "" and "title" not in str(shot.get("effect", "")):
            pass
    # Title special handling
    if "漫剧" in text or "AI" in text:
        for frame in frames:
            add_title(frame, text)
    
    return frames


def generate_voice(lines, output_mp3):
    """Generate edge-tts voiceover."""
    text = " ".join(lines)
    if not text.strip():
        return None
    cmd = [
        sys.executable, "-m", "edge_tts",
        "--voice", "zh-CN-XiaoxiaoNeural",
        "--text", text,
        "--write-media", output_mp3,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_mp3


def assemble(frames_list, output_mp4, audio_mp3=None):
    """Write all frames to a silent video, then mux with audio."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    silent = OUT_DIR / f"manhua-silent-{stamp}.mp4"
    final = OUT_DIR / f"manhua-drama-{stamp}.mp4"

    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for frames in frames_list:
        for frame in frames:
            writer.write(frame)
    writer.release()

    if audio_mp3 and os.path.exists(audio_mp3):
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(silent),
            "-i", audio_mp3,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", str(final),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(silent),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(final),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return final


def main():
    print("=== Motion Comic Assembly ===")

    # Render all shots
    all_frames = []
    for i, shot in enumerate(SHOTS):
        print(f"Shot {i+1}: {shot.get('text', '')} ({shot['type']})")
        frames = render_shot(shot, FPS, WIDTH, HEIGHT)
        all_frames.append(frames)
        print(f"  → {len(frames)} frames ({len(frames)/FPS:.1f}s)")

    # Generate voiceover
    voice_lines = [s["voice"] for s in SHOTS if s["voice"]]
    audio_path = None
    if voice_lines:
        audio_path = OUT_DIR / "manhua-voiceover.mp3"
        try:
            print("Generating voiceover...")
            generate_voice(voice_lines, str(audio_path))
        except Exception as e:
            print(f"Voice skipped: {e}")
            audio_path = None
    
    # Assemble final video
    print("Assembling final video...")
    final = assemble(all_frames, str(OUT_DIR / "manhua-final.mp4"), str(audio_path) if audio_path else None)
    print(f"Done: {final}")


if __name__ == "__main__":
    main()
