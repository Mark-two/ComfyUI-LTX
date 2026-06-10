#!/usr/bin/env python3
"""Download LightX2V FP8 scaled ComfyUI models."""
from huggingface_hub import hf_hub_download

JOBS = [
    ("lightx2v/Wan2.2-Distill-Models",
     "wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors",
     "/home/kang/Documents/ai-video-local/models/wan22-fp8"),
    ("lightx2v/Wan2.2-Distill-Models",
     "wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors",
     "/home/kang/Documents/ai-video-local/models/wan22-fp8"),
]

for repo, filename, local_dir in JOBS:
    print(f"DOWNLOAD {filename}", flush=True)
    path = hf_hub_download(repo, filename, local_dir=local_dir)
    print(f"DONE {path}", flush=True)
print("ALL DONE", flush=True)
