#!/usr/bin/env python3
import math
import time
from pathlib import Path

import cv2
import numpy as np


SOURCE = Path("/home/kang/Documents/novel-to-video/outputs/3d-manhua-heroine-diverse/20260607-201952-01.png")
OUT_DIR = Path("/home/kang/Documents/ComfyUI/output/local_manhua")


def resize_cover(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(width / w, height / h)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    y = (resized.shape[0] - height) // 2
    x = (resized.shape[1] - width) // 2
    return resized[y:y + height, x:x + width]


def composite_character(bg: np.ndarray, char: np.ndarray, x: int, y: int) -> np.ndarray:
    out = bg.copy()
    h, w = char.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(out.shape[1], x + w), min(out.shape[0], y + h)
    if x1 >= x2 or y1 >= y2:
        return out
    crop = char[y1 - y:y2 - y, x1 - x:x2 - x]
    # Soft oval matte keeps the full image usable while separating the character from background.
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 2, int(h * 0.50)), (int(w * 0.46), int(h * 0.52)), 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 18)
    m = mask[y1 - y:y2 - y, x1 - x:x2 - x, None]
    out[y1:y2, x1:x2] = (crop * m + out[y1:y2, x1:x2] * (1 - m)).astype(np.uint8)
    return out


def add_speed_lines(frame: np.ndarray, strength: float) -> np.ndarray:
    h, w = frame.shape[:2]
    layer = frame.copy()
    center = (w // 2, int(h * 0.40))
    for i in range(36):
        ang = i / 36 * math.tau
        r1 = int(80 + 80 * strength)
        r2 = int(430 + 160 * strength)
        p1 = (int(center[0] + math.cos(ang) * r1), int(center[1] + math.sin(ang) * r1))
        p2 = (int(center[0] + math.cos(ang) * r2), int(center[1] + math.sin(ang) * r2))
        cv2.line(layer, p1, p2, (230, 245, 255), 2, cv2.LINE_AA)
    return cv2.addWeighted(layer, 0.18 * strength, frame, 1 - 0.18 * strength, 0)


def main() -> None:
    src = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)
    if src is None:
        raise SystemExit(f"missing source: {SOURCE}")

    width, height = 512, 768
    src = resize_cover(src, width, height)
    bg = cv2.GaussianBlur(src, (0, 0), 18)
    bg = cv2.addWeighted(bg, 0.72, np.full_like(bg, (35, 24, 55)), 0.28, 0)

    fps = 24
    frames = 72
    stamp = time.strftime("%Y%m%d-%H%M%S")
    silent = OUT_DIR / f"jump-test-silent-{stamp}.mp4"
    final = OUT_DIR / f"jump-test-{stamp}.mp4"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for i in range(frames):
        t = i / (frames - 1)
        # Parabolic jump: crouch -> launch -> apex -> landing.
        jump = 4 * t * (1 - t)
        crouch = max(0, 1 - abs(t - 0.08) / 0.08)
        land = max(0, 1 - abs(t - 0.88) / 0.07)
        y_offset = int(210 * jump - 30 * crouch + 22 * land)
        scale_y = 1.0 - 0.08 * crouch + 0.05 * land
        scale_x = 1.0 + 0.06 * crouch - 0.03 * land
        angle = 3.5 * math.sin(t * math.pi) * math.sin(t * math.tau * 1.2)

        frame = bg.copy()
        if jump > 0.12:
            frame = add_speed_lines(frame, min(1.0, jump * 1.4))

        m = cv2.getRotationMatrix2D((width / 2, height * 0.52), angle, 1.0)
        warped = cv2.warpAffine(src, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        char = cv2.resize(warped, (int(width * scale_x), int(height * scale_y)), interpolation=cv2.INTER_CUBIC)
        x = (width - char.shape[1]) // 2
        y = int((height - char.shape[0]) // 2 - y_offset)

        shadow_w = int(220 * (1.05 - 0.65 * jump))
        shadow_h = int(28 * (1.05 - 0.45 * jump))
        cv2.ellipse(frame, (width // 2, height - 58), (max(20, shadow_w), max(6, shadow_h)), 0, 0, 360, (8, 8, 16), -1, cv2.LINE_AA)
        frame = composite_character(frame, char, x, y)

        if land > 0.25:
            shake = int(10 * land)
            mat = np.float32([[1, 0, math.sin(i) * shake], [0, 1, math.cos(i * 1.3) * shake]])
            frame = cv2.warpAffine(frame, mat, (width, height), borderMode=cv2.BORDER_REFLECT)
            cv2.circle(frame, (width // 2, height - 62), int(120 * land), (210, 230, 255), 3, cv2.LINE_AA)

        cv2.putText(frame, "JUMP!", (32, 70), cv2.FONT_HERSHEY_DUPLEX, 1.25, (255, 240, 220), 2, cv2.LINE_AA)
        writer.write(frame)

    writer.release()

    import subprocess
    duration = frames / fps
    cmd = [
        "ffmpeg", "-y",
        "-i", str(silent),
        "-f", "lavfi", "-t", f"{duration:.3f}",
        "-i", "sine=frequency=72:sample_rate=44100",
        "-filter_complex", "[1:a]volume=0.09[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-shortest", str(final),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(final)


if __name__ == "__main__":
    main()
