#!/usr/bin/env python3
"""Download Wan2.2 Q5_K_M 720p GGUF models"""
from huggingface_hub import hf_hub_download

JOBS = [
    ("jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF",
     "high_noise_260412/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412-Q5_K_M.gguf",
     "/home/kang/Documents/ai-video-local/models/wan22-i2v-gguf/high_noise_260412"),
    ("jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF",
     "low_noise_260412/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412-Q5_K_M.gguf",
     "/home/kang/Documents/ai-video-local/models/wan22-i2v-gguf/low_noise_260412"),
]

for repo, filename, local_dir in JOBS:
    print(f"DOWNLOAD {filename}", flush=True)
    path = hf_hub_download(repo, filename, local_dir=local_dir)
    print(f"DONE {path}", flush=True)

print("ALL DONE", flush=True)
