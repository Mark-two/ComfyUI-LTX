#!/usr/bin/env python3
from huggingface_hub import hf_hub_download

print("DOWNLOAD umt5_xxl_fp16.safetensors", flush=True)
path = hf_hub_download(
    "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
    "split_files/text_encoders/umt5_xxl_fp16.safetensors",
    local_dir="/home/kang/Documents/ai-video-local/models",
)
print(f"DONE {path}", flush=True)
