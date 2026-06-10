#!/usr/bin/env python3
from huggingface_hub import hf_hub_download


JOBS = [
    (
        "jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF",
        "high_noise_260412/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412-Q3_K_S.gguf",
        "/home/kang/Documents/ai-video-local/models/wan22-i2v-gguf/high_noise_260412",
    ),
    (
        "jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF",
        "low_noise_260412/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412-Q3_K_S.gguf",
        "/home/kang/Documents/ai-video-local/models/wan22-i2v-gguf/low_noise_260412",
    ),
    (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "/home/kang/Documents/ai-video-local/models",
    ),
    (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "split_files/vae/wan2.2_vae.safetensors",
        "/home/kang/Documents/ai-video-local/models",
    ),
]


for repo, filename, local_dir in JOBS:
    print(f"DOWNLOAD {repo} {filename} -> {local_dir}", flush=True)
    path = hf_hub_download(repo, filename, local_dir=local_dir)
    print(f"DONE {path}", flush=True)

print("ALL DONE", flush=True)
