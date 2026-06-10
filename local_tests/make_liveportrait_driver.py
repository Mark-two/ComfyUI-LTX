#!/usr/bin/env python3
import math
import time
from pathlib import Path

import cv2
import numpy as np


INPUT = Path("/home/kang/Documents/ComfyUI/output/local_manhua/majicmix_keyframe_00001_.png")
OUT_DIR = Path("/home/kang/Documents/ComfyUI/output/local_manhua")


def resize_cover(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(width / w, height / h)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    y = (resized.shape[0] - height) // 2
    x = (resized.shape[1] - width) // 2
    return resized[y:y + height, x:x + width]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = cv2.imread(str(INPUT), cv2.IMREAD_COLOR)
    if src is None:
        raise SystemExit(f"missing source: {INPUT}")

    width = height = 512
    src = resize_cover(src, width, height)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = OUT_DIR / f"liveportrait-driver-{stamp}.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 16, (width, height))

    # Face feature estimates for the generated portrait crop.
    le = (width * 0.43, height * 0.34)
    re = (width * 0.57, height * 0.34)
    mouth = (width * 0.50, height * 0.47)

    for i in range(64):
        t = i / 63
        yaw = 9.0 * math.sin(t * math.tau)
        pitch = 5.0 * math.sin(t * math.tau * 0.55 - 0.3)
        scale_x = 1.0 + 0.045 * math.sin(t * math.tau)
        scale_y = 1.0 + 0.025 * math.sin(t * math.tau * 0.7)
        m = cv2.getRotationMatrix2D((width / 2, height * 0.42), yaw, 1.0)
        m[0, 0] *= scale_x
        m[0, 1] *= scale_x
        m[1, 0] *= scale_y
        m[1, 1] *= scale_y
        m[1, 2] += pitch
        frame = cv2.warpAffine(src, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        # Add clear expression signal for LivePortrait landmark extraction.
        talk = max(0.0, math.sin(t * math.tau * 3.2))
        if talk > 0.15:
            cv2.ellipse(frame, (int(mouth[0]), int(mouth[1] + 6)), (26, int(8 + 20 * talk)), 0, 0, 360, (18, 12, 18), -1, cv2.LINE_AA)

        blink = max(0.0, 1.0 - abs(t - 0.30) / 0.045) + max(0.0, 1.0 - abs(t - 0.72) / 0.045)
        if blink > 0:
            for eye in (le, re):
                cv2.ellipse(frame, (int(eye[0]), int(eye[1])), (25, int(5 + 11 * min(1.0, blink))), 0, 0, 360, (170, 145, 135), -1, cv2.LINE_AA)
                cv2.line(frame, (int(eye[0] - 20), int(eye[1])), (int(eye[0] + 20), int(eye[1])), (30, 25, 30), 2, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print(out)


if __name__ == "__main__":
    main()
