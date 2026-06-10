#!/usr/bin/env python3
"""Generate a synthetic OpenPose skeleton animation showing a jump sequence."""
import math
import time
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("/home/kang/Documents/ComfyUI/input/jump_skeleton")
WIDTH, HEIGHT = 512, 768
FPS = 12
FRAMES = 36  # 3 seconds


SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # nose->neck->shoulders->elbows
    (1, 8),                                    # neck to hip center
    (8, 9), (8, 12), (9, 10), (10, 11),        # hip->legs
    (12, 13), (13, 14),
    (1, 5), (5, 6), (6, 7),                    # right arm
    (1, 2), (2, 3), (3, 4),                    # left arm
]


def get_pose(t):
    """Return list of (x, y, confidence) keypoints for a jump at normalized time t [0, 1]."""
    cx, _ = WIDTH / 2, HEIGHT / 2
    jump = 4 * t * (1 - t)
    crouch = max(0, 1 - abs(t - 0.08) / 0.08)
    land = max(0, 1 - abs(t - 0.85) / 0.07)

    base_y = 480
    y_shift = int(260 * jump - 40 * crouch + 30 * land)

    kp = np.zeros((18, 3), dtype=np.float32)
    kp[:, 2] = 1.0  # confidence

    # Coordinates in (x, y) format, normalized for a standing person
    nose = (cx, int(base_y - 520 - y_shift))
    neck = (cx, int(base_y - 450 - y_shift))
    l_shoulder = (int(cx - 55), int(base_y - 440 - y_shift))
    r_shoulder = (int(cx + 55), int(base_y - 440 - y_shift))
    l_elbow = (int(cx - 95 + 15 * crouch), int(base_y - 370 - y_shift))
    r_elbow = (int(cx + 95 - 15 * crouch), int(base_y - 370 - y_shift))
    l_wrist = (int(cx - 120), int(base_y - 290 - y_shift + 40 * jump))
    r_wrist = (int(cx + 120), int(base_y - 290 - y_shift + 40 * jump))
    l_hip = (int(cx - 30), int(base_y - 210 - y_shift))
    r_hip = (int(cx + 30), int(base_y - 210 - y_shift))
    l_knee = (int(cx - 40), int(base_y - 100 - y_shift - 30 * crouch))
    r_knee = (int(cx + 40), int(base_y - 100 - y_shift - 30 * crouch))
    l_ankle = (int(cx - 25), int(base_y + 20 - y_shift + 10 * crouch))
    r_ankle = (int(cx + 25), int(base_y + 20 - y_shift + 10 * crouch))
    l_eye = (int(cx - 14), int(base_y - 530 - y_shift))
    r_eye = (int(cx + 14), int(base_y - 530 - y_shift))
    l_ear = (int(cx - 30), int(base_y - 510 - y_shift))
    r_ear = (int(cx + 30), int(base_y - 510 - y_shift))

    kp[0] = (nose[0], nose[1], 1)        # nose
    kp[1] = (neck[0], neck[1], 1)        # neck
    kp[2] = (r_shoulder[0], r_shoulder[1], 1)  # R shoulder
    kp[3] = (r_elbow[0], r_elbow[1], 1)  # R elbow
    kp[4] = (r_wrist[0], r_wrist[1], 1)  # R wrist
    kp[5] = (l_shoulder[0], l_shoulder[1], 1)  # L shoulder
    kp[6] = (l_elbow[0], l_elbow[1], 1)  # L elbow
    kp[7] = (l_wrist[0], l_wrist[1], 1)  # L wrist
    kp[8] = (cx, int(base_y - 240 - y_shift), 1)  # mid hip
    kp[9] = (r_hip[0], r_hip[1], 1)      # R hip
    kp[10] = (r_knee[0], r_knee[1], 1)   # R knee
    kp[11] = (r_ankle[0], r_ankle[1], 1) # R ankle
    kp[12] = (l_hip[0], l_hip[1], 1)     # L hip
    kp[13] = (l_knee[0], l_knee[1], 1)   # L knee
    kp[14] = (l_ankle[0], l_ankle[1], 1) # L ankle
    kp[15] = (r_eye[0], r_eye[1], 1)     # R eye
    kp[16] = (l_eye[0], l_eye[1], 1)     # L eye
    kp[17] = (r_ear[0], r_ear[1], 1)     # R ear

    return kp


def draw_skeleton(img, kps, pairs=SKELETON_PAIRS):
    """Draw OpenPose-style skeleton on image."""
    h, w = img.shape[:2]
    # Draw bones
    limb_colors = [
        (80, 200, 80),   # neck
        (80, 200, 80),   # spine
        (60, 160, 255),  # right arm (blue)
        (60, 160, 255),
        (60, 160, 255),
        (255, 100, 60),  # left arm (red)
        (255, 100, 60),
        (255, 100, 60),
        (80, 200, 80),   # hip
        (100, 220, 140), # right leg
        (100, 220, 140),
        (160, 100, 240), # left leg
        (160, 100, 240),
        (160, 100, 240),
    ]
    for (a, b), color in zip(pairs, limb_colors):
        if a < len(kps) and b < len(kps):
            pt1 = (int(kps[a][0]), int(kps[a][1]))
            pt2 = (int(kps[b][0]), int(kps[b][1]))
            if all(0 <= c < lim for c, lim in zip(pt1 + pt2, (w, h, w, h))):
                cv2.line(img, pt1, pt2, color, 3, cv2.LINE_AA)

    # Draw keypoints
    for i, (x, y, conf) in enumerate(kps):
        if conf < 0.3:
            continue
        if i in (0, 15, 16):
            color = (255, 255, 255)
            r = 5
        elif i in (4, 7):
            color = (0, 255, 255)
            r = 4
        elif i in (11, 14):
            color = (255, 0, 255)
            r = 4
        else:
            color = (0, 220, 255)
            r = 3
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (int(x), int(y)), r, color, -1, cv2.LINE_AA)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = OUT_DIR / f"jump-skeleton-{stamp}.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))

    for i in range(FRAMES):
        t = i / (FRAMES - 1)
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        kps = get_pose(t)
        draw_skeleton(img, kps)
        writer.write(img)

    writer.release()
    print(out)

    # Also save individual frames for ComfyUI
    frames_dir = OUT_DIR / stamp
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(out))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(frames_dir / f"frame_{idx:04d}.png"), frame)
        idx += 1
    cap.release()
    print(f"frames: {frames_dir} ({idx} frames)")


if __name__ == "__main__":
    main()
