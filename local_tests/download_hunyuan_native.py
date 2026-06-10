#!/usr/bin/env python3
from huggingface_hub import hf_hub_download


JOBS = [
    (
        "Kijai/HunyuanVideo_comfy",
        "hunyuan_video_I2V_fp8_e4m3fn.safetensors",
        "/home/kang/Documents/ai-video-local/models/hyvideo",
    ),
    (
        "Kijai/HunyuanVideo_comfy",
        "hunyuan_video_vae_bf16.safetensors",
        "/home/kang/Documents/ai-video-local/models/hyvid",
    ),
    (
        "Comfy-Org/HunyuanVideo_repackaged",
        "split_files/text_encoders/clip_l.safetensors",
        "/home/kang/Documents/ai-video-local/models",
    ),
    (
        "Comfy-Org/HunyuanVideo_repackaged",
        "split_files/text_encoders/llava_llama3_fp16.safetensors",
        "/home/kang/Documents/ai-video-local/models",
    ),
    (
        "Comfy-Org/HunyuanVideo_repackaged",
        "split_files/clip_vision/llava_llama3_vision.safetensors",
        "/home/kang/Documents/ai-video-local/models",
    ),
]


for repo, filename, local_dir in JOBS:
    print(f"DOWNLOAD {repo} {filename} -> {local_dir}", flush=True)
    path = hf_hub_download(repo, filename, local_dir=local_dir)
    print(f"DONE {path}", flush=True)

print("ALL DONE", flush=True)
