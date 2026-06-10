#!/usr/bin/env python3
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


INPUT = Path("/home/kang/Documents/ComfyUI/output/local_manhua/majicmix_keyframe_00001_.png")
OUT_DIR = Path("/home/kang/Documents/ComfyUI/output/local_manhua")


def run(cmd: list[str]) -> None:
    print("$", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def resize_cover(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(width / w, height / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    x = (nw - width) // 2
    y = (nh - height) // 2
    return resized[y : y + height, x : x + width]


def alpha_blend(base: np.ndarray, overlay: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha3 = alpha[..., None]
    return (base * (1.0 - alpha3) + overlay * alpha3).astype(np.uint8)


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"missing input: {INPUT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    silent = OUT_DIR / f"visible-motion-{stamp}-silent.mp4"
    output = OUT_DIR / f"visible-motion-{stamp}.mp4"

    width, height = 480, 832
    fps = 24
    duration = 5.0
    total = int(fps * duration)

    src = cv2.imread(str(INPUT), cv2.IMREAD_COLOR)
    if src is None:
        raise SystemExit(f"failed to read: {INPUT}")
    src = resize_cover(src, width, height)

    # Soft portrait masks. Keep the face stable, move the surrounding image more.
    yy, xx = np.mgrid[0:height, 0:width]
    face_mask = np.exp(-(((xx - width * 0.50) / 115) ** 2 + ((yy - height * 0.31) / 130) ** 2)).astype(np.float32)
    body_mask = np.exp(-(((xx - width * 0.50) / 180) ** 2 + ((yy - height * 0.62) / 260) ** 2)).astype(np.float32)
    bg_mask = np.clip(1.0 - np.maximum(face_mask * 1.3, body_mask * 0.55), 0.0, 1.0)
    lower_mask = np.clip((yy - height * 0.45) / (height * 0.35), 0.0, 1.0).astype(np.float32)

    writer = cv2.VideoWriter(
        str(silent),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for i in range(total):
        t = i / (total - 1)
        wave = math.sin(t * math.tau)
        wave2 = math.sin(t * math.tau * 2.0 + 0.6)

        # Stronger camera move: slow push-in plus small horizontal drift.
        scale = 1.0 + 0.085 * t
        dx = int(12 * math.sin(t * math.tau * 0.65))
        dy = int(-18 * t + 4 * wave)
        m = cv2.getRotationMatrix2D((width / 2, height / 2), 0.9 * wave, scale)
        m[0, 2] += dx
        m[1, 2] += dy
        cam = cv2.warpAffine(src, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        # Background has a separate parallax shift, visibly moving city lights.
        bg_m = np.float32([[1, 0, -24 * t + 8 * wave], [0, 1, 8 * wave2]])
        bg = cv2.warpAffine(cam, bg_m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        frame = alpha_blend(cam, bg, bg_mask * 0.75)

        # Subtle cloth/body breathing: lower part shifts independently.
        cloth_m = np.float32([[1, 0, 3.5 * wave], [0, 1, 5.5 * math.sin(t * math.tau * 1.4)]])
        cloth = cv2.warpAffine(frame, cloth_m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        frame = alpha_blend(frame, cloth, lower_mask * body_mask * 0.45)

        # Moving rim light sweep.
        sweep_center = int(width * (-0.25 + 1.5 * t))
        sweep = np.exp(-((xx - sweep_center) / 70) ** 2).astype(np.float32) * 0.32
        blue = np.zeros_like(frame)
        blue[:, :, 0] = 80
        blue[:, :, 1] = 35
        frame = np.clip(frame.astype(np.float32) + blue.astype(np.float32) * sweep[..., None], 0, 255).astype(np.uint8)

        # Cinematic particles, deterministic per frame.
        rng = np.random.default_rng(260607 + i)
        particles = np.zeros_like(frame)
        for _ in range(22):
            px = int((rng.integers(0, width) + 55 * t) % width)
            py = int((rng.integers(0, height) - 90 * t) % height)
            radius = int(rng.integers(1, 3))
            color = (int(rng.integers(120, 210)), int(rng.integers(120, 190)), int(rng.integers(160, 255)))
            cv2.circle(particles, (px, py), radius, color, -1, lineType=cv2.LINE_AA)
        particles = cv2.GaussianBlur(particles, (0, 0), 1.2)
        frame = cv2.addWeighted(frame, 1.0, particles, 0.55, 0)

        # Brief blink-like shadow over eye area. It is intentionally subtle to avoid face damage.
        blink = max(0.0, 1.0 - abs(t - 0.42) / 0.035) + max(0.0, 1.0 - abs(t - 0.78) / 0.03)
        if blink > 0:
            eye_alpha = np.exp(-(((xx - width * 0.50) / 92) ** 2 + ((yy - height * 0.285) / 22) ** 2)).astype(np.float32)
            shadow = (frame.astype(np.float32) * 0.55).astype(np.uint8)
            frame = alpha_blend(frame, shadow, np.clip(eye_alpha * blink * 0.42, 0, 0.42))

        # Final contrast and vignette.
        frame = cv2.convertScaleAbs(frame, alpha=1.06, beta=2)
        dist = ((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2
        vignette = np.clip(1.0 - 0.34 * dist, 0.62, 1.0).astype(np.float32)
        frame = (frame.astype(np.float32) * vignette[..., None]).astype(np.uint8)

        writer.write(frame)

    writer.release()

    run([
        "ffmpeg", "-y",
        "-i", str(silent),
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(output),
    ])
    print(output)


if __name__ == "__main__":
    main()
