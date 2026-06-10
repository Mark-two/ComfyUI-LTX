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


def smooth_mask(width: int, height: int, center: tuple[float, float], rx: float, ry: float, power: float = 1.0) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    d = ((xx - center[0]) / rx) ** 2 + ((yy - center[1]) / ry) ** 2
    return np.clip(np.exp(-d * power), 0, 1).astype(np.float32)


def blend(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask[..., None]
    return np.clip(base.astype(np.float32) * (1 - m) + overlay.astype(np.float32) * m, 0, 255).astype(np.uint8)


def poly_center(points: list[tuple[float, float]]) -> tuple[float, float]:
    arr = np.array(points, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def skin_color(img: np.ndarray, center: tuple[float, float], radius: int = 18) -> tuple[int, int, int]:
    x, y = int(center[0]), int(center[1])
    h, w = img.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius)
    y0, y1 = max(0, y - radius), min(h, y + radius)
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return (190, 170, 155)
    med = np.median(patch.reshape(-1, 3), axis=0)
    return tuple(int(v) for v in med)


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"missing input: {INPUT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    silent = OUT_DIR / f"local-face-motion-{stamp}-silent.mp4"
    output = OUT_DIR / f"local-face-motion-{stamp}.mp4"

    width, height = 480, 832
    fps = 24
    duration = 5.0
    frames = int(fps * duration)

    src0 = cv2.imread(str(INPUT), cv2.IMREAD_COLOR)
    if src0 is None:
        raise SystemExit(f"failed to read: {INPUT}")
    src = resize_cover(src0, width, height)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(80, 80))
    if len(faces):
        # Pick the largest detected face and derive approximate feature positions.
        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        le = (x + fw * 0.34, y + fh * 0.42)
        re = (x + fw * 0.66, y + fh * 0.42)
        mc = (x + fw * 0.50, y + fh * 0.72)
        jawc = (x + fw * 0.50, y + fh * 1.02)
        eye_dist = max(60.0, abs(re[0] - le[0]))
        mouth_w = max(44.0, fw * 0.36)
    else:
        # Stable fallback for the generated 480x832 upper-body portrait.
        le = (width * 0.43, height * 0.285)
        re = (width * 0.57, height * 0.285)
        mc = (width * 0.50, height * 0.365)
        jawc = (width * 0.50, height * 0.455)
        eye_dist = width * 0.15
        mouth_w = width * 0.16

    face_c = ((le[0] + re[0] + mc[0]) / 3, (le[1] + re[1] + mc[1]) / 3)
    face_mask = smooth_mask(width, height, face_c, eye_dist * 1.12, eye_dist * 1.55, 0.9)
    mouth_mask = smooth_mask(width, height, mc, mouth_w * 0.72, mouth_w * 0.38, 1.4)
    jaw_mask = smooth_mask(width, height, jawc, eye_dist * 1.0, eye_dist * 0.75, 1.0)

    # Side hair masks around both cheeks; the face center remains locked.
    hair_left = smooth_mask(width, height, (face_c[0] - eye_dist * 0.78, face_c[1] + eye_dist * 0.35), eye_dist * 0.35, eye_dist * 1.35, 0.8)
    hair_right = smooth_mask(width, height, (face_c[0] + eye_dist * 0.78, face_c[1] + eye_dist * 0.35), eye_dist * 0.35, eye_dist * 1.35, 0.8)
    hair_mask = np.clip((hair_left + hair_right) * (1.0 - face_mask * 0.65), 0, 1)

    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    skin = skin_color(src, ((le[0] + re[0]) / 2, (le[1] + re[1]) / 2 + 12))
    lip_color = skin_color(src, mc, 10)

    for i in range(frames):
        t = i / (frames - 1)
        frame = src.copy()

        # No global drift. Only tiny face micro-expression within the face mask.
        nod = 1.8 * math.sin(t * math.tau * 0.8)
        face_m = np.float32([[1, 0, 0.7 * math.sin(t * math.tau * 0.55)], [0, 1, nod]])
        face_shift = cv2.warpAffine(src, face_m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        frame = blend(frame, face_shift, face_mask * 0.22)

        # Shoulder/chin breathing, localized below the mouth. This should not look like floating.
        breath = 4.0 * max(0.0, math.sin(t * math.tau * 1.1))
        jaw_m = np.float32([[1, 0, 0], [0, 1, breath]])
        jaw_shift = cv2.warpAffine(frame, jaw_m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        frame = blend(frame, jaw_shift, jaw_mask * 0.20)

        # Hair side movement only, alternating left/right.
        hair_dx = 4.0 * math.sin(t * math.tau * 1.3)
        hair_m = np.float32([[1, 0, hair_dx], [0, 1, 1.5 * math.sin(t * math.tau * 0.9)]])
        hair_shift = cv2.warpAffine(frame, hair_m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        frame = blend(frame, hair_shift, hair_mask * 0.42)

        # Mouth speaking: visible dark opening plus lip compression, tied to syllable-like beats.
        talk = 0.5 + 0.5 * math.sin(t * math.tau * 4.2)
        talk *= 0.65 + 0.35 * math.sin(t * math.tau * 1.1 + 0.4)
        mouth_overlay = frame.copy()
        open_h = int(3 + 12 * max(0, talk))
        cv2.ellipse(
            mouth_overlay,
            (int(mc[0]), int(mc[1] + 2)),
            (int(mouth_w * 0.28), open_h),
            0,
            0,
            360,
            (24, 18, 22),
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.ellipse(
            mouth_overlay,
            (int(mc[0]), int(mc[1] - open_h * 0.52)),
            (int(mouth_w * 0.35), max(2, open_h // 4)),
            0,
            0,
            180,
            tuple(int(v) for v in lip_color),
            2,
            lineType=cv2.LINE_AA,
        )
        frame = blend(frame, mouth_overlay, mouth_mask * 0.86)

        # Blink: paint eyelid-colored ellipses over each eye, timed as actual short blinks.
        blink = max(0.0, 1.0 - abs(t - 0.26) / 0.032) + max(0.0, 1.0 - abs(t - 0.67) / 0.036)
        if blink > 0:
            eye_overlay = frame.copy()
            for ec in (le, re):
                rx = int(eye_dist * 0.18)
                ry = int(5 + 12 * min(1.0, blink))
                cv2.ellipse(
                    eye_overlay,
                    (int(ec[0]), int(ec[1])),
                    (rx, ry),
                    0,
                    0,
                    360,
                    skin,
                    -1,
                    lineType=cv2.LINE_AA,
                )
                cv2.line(
                    eye_overlay,
                    (int(ec[0] - rx * 0.82), int(ec[1] + ry * 0.12)),
                    (int(ec[0] + rx * 0.82), int(ec[1] + ry * 0.06)),
                    (35, 30, 35),
                    1,
                    lineType=cv2.LINE_AA,
                )
            frame = cv2.addWeighted(frame, 1 - min(1.0, blink) * 0.86, eye_overlay, min(1.0, blink) * 0.86, 0)

        # Eye sparkle shift so the gaze feels alive.
        sparkle = frame.copy()
        for ec in (le, re):
            sx = int(ec[0] + 8 + 1.5 * math.sin(t * math.tau * 1.7))
            sy = int(ec[1] - 5 + 0.8 * math.cos(t * math.tau * 1.3))
            cv2.circle(sparkle, (sx, sy), 2, (245, 245, 255), -1, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(frame, 0.98, sparkle, 0.02, 0)

        # Keep background locked; only a mild static vignette/color grade.
        frame = cv2.convertScaleAbs(frame, alpha=1.035, beta=1)
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
