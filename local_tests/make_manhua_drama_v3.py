#!/usr/bin/env python3
"""Manhua drama assembler v3: pure ffmpeg, no OpenCV re-encoding."""
import subprocess
import time
from pathlib import Path

OUT = Path("/home/kang/Documents/ComfyUI/output/local_manhua")
W, H = 1080, 1920

SHOTS = [
    {"src": OUT / "manhua-shot1_00001_smooth.mp4", "text": "夜色天台，她独自伫立"},
    {"src": OUT / "liveportrait-test-manhua_closeup_00001_-d0-d0.45-e0.45-l0.45_00001.mp4", "text": "眼神坚定"},
    {"src": OUT / "manhua-shot2_00001_smooth.mp4", "text": "每一步都是力量"},
    {"src": OUT / "manhua-shot3_00001_smooth.mp4", "text": "转身，等待命运"},
]


def write_concat_file(shots, tmpdir):
    """Write ffmpeg concat demuxer file."""
    concat_path = tmpdir / "concat.txt"
    with open(concat_path, "w") as f:
        for shot in shots:
            src = Path(shot["src"])
            if not src.exists():
                print(f"WARNING: missing {src}")
                continue
            f.write(f"file '{src.absolute()}'\n")
    return concat_path


def main():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tmp = OUT / f"manhua-tmp-{stamp}"
    tmp.mkdir(exist_ok=True)

    # Step 1: Build subtitle filter for each segment
    # We'll use drawtext filter with per-segment timing
    filters = []
    total_dur = 0
    seg_i = 0
    
    for shot in SHOTS:
        src = Path(shot["src"])
        if not src.exists():
            continue
        # Get video duration
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(src)
        ], capture_output=True, text=True, check=True)
        dur = float(probe.stdout.strip())
        
        filters.append(
            f"[{seg_i}:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},"
            f"unsharp=7:7:1.5:7:7:0.5,"  # sharpen to fix I2V softness
            f"drawtext=text='{shot['text']}':fontsize=48:fontcolor=white@0.9:"
            f"x=(w-text_w)/2:y=h-120:box=1:boxcolor=black@0.4:boxborderw=10:"
            f"enable='between(t,0,{dur})'[v{seg_i}]"
        )
        total_dur += dur
        seg_i += 1

    # Step 2: Build ffmpeg command
    inputs = []
    for shot in SHOTS:
        src = Path(shot["src"])
        if src.exists():
            inputs.extend(["-i", str(src.absolute())])

    filter_complex = ";".join(filters)
    concat_parts = "".join(f"[v{i}]" for i in range(len(SHOTS)))
    filter_complex += f";{concat_parts}concat=n={len(SHOTS)}:v=1[outv]"

    final = OUT / f"manhua-drama-v3-{stamp}.mp4"
    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-r", "24",
        str(final),
    ]
    
    print("Running ffmpeg...")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    print(f"Done: {final}")


if __name__ == "__main__":
    main()
