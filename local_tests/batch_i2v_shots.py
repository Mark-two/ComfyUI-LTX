#!/usr/bin/env python3
"""Batch Wan2.2 I2V generator for manhua drama shots."""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER = os.environ.get("COMFY_SERVER", "http://127.0.0.1:8188")
OUTPUT_DIR = Path(os.environ.get("COMFY_OUTPUT", os.path.expanduser("~/Documents/ComfyUI/output")))

SHOTS = [
    {
        "source": "/home/kang/Documents/ComfyUI/output/local_manhua/heroine-hq_00001_.png",
        "prompt": "A beautiful young woman in cream blouse and black pencil skirt, standing confidently on a city rooftop at golden hour, hair gently swaying in warm breeze, subtle natural body movement, calm breathing, cinematic lighting, sharp clean image",
        "neg": "jerky motion, jumping, bouncing, running, fast movement, static frame, frozen, low quality, blurry, distorted face, deformed body, ugly, text, watermark, grain, noise",
        "w": 480, "h": 832, "frames": 33, "fps": 16,
        "prefix": "local_manhua/manhua-shot1",
    },
    {
        "source": "/home/kang/Documents/ComfyUI/output/local_manhua/heroine-hq_00002_.png",
        "prompt": "A beautiful young woman in cream blouse and black pencil skirt, walking forward with natural slow strides on a city rooftop at golden hour, legs stepping one after another, arms swinging, hair flowing, smooth continuous motion, cinematic warm sunset lighting",
        "neg": "static, still, frozen, teleporting, glitch, low quality, blurry, distorted face, deformed, ugly, text, watermark, grain, noise",
        "w": 480, "h": 832, "frames": 49, "fps": 16,
        "prefix": "local_manhua/manhua-shot2",
    },
    {
        "source": "/home/kang/Documents/ComfyUI/output/local_manhua/heroine-hq_00003_.png",
        "prompt": "A beautiful young woman in cream blouse and black pencil skirt, turning slightly to look over her shoulder on a city rooftop at golden hour, elegant slow movement, hair flowing, confident expression, cinematic warm lighting, sharp clean image",
        "neg": "static, still, frozen, jerky, fast, low quality, blurry, distorted, ugly, text, watermark, grain, noise",
        "w": 480, "h": 832, "frames": 33, "fps": 16,
        "prefix": "local_manhua/manhua-shot3",
    },
]

CHECKPOINT = "majicmixRealistic_v7.safetensors"
T5_MODEL = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_MODEL = "split_files/vae/wan_2.1_vae.safetensors"
HIGH_MODEL = "wan22-i2v-gguf/high_noise_260412/high_noise_260412/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412-Q5_K_M.gguf"
LOW_MODEL = "wan22-i2v-gguf/low_noise_260412/low_noise_260412/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412-Q5_K_M.gguf"


def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}\n{body}") from exc


def get_json(path):
    with urllib.request.urlopen(SERVER + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_for_prompt(prompt_id, timeout_seconds=3600):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        history = get_json(f"/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(3)
    raise TimeoutError(f"Timed out {prompt_id}")


def generate_shot(shot_info, seed):
    wf = {
        "1": {"class_type": "VHS_LoadImagePath", "inputs": {"image": shot_info["source"], "custom_width": shot_info["w"], "custom_height": shot_info["h"]}},
        "2": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": VAE_MODEL, "precision": "bf16"}},
        "3": {"class_type": "LoadWanVideoT5TextEncoder", "inputs": {"model_name": T5_MODEL, "precision": "bf16", "load_device": "offload_device", "quantization": "disabled"}},
        "4": {"class_type": "WanVideoImageToVideoEncode", "inputs": {"width": shot_info["w"], "height": shot_info["h"], "num_frames": shot_info["frames"], "noise_aug_strength": 0.0, "start_latent_strength": 0.9, "end_latent_strength": 1.15, "force_offload": True, "vae": ["2", 0], "start_image": ["1", 0], "fun_or_fl2v_model": False, "tiled_vae": True}},
        "5": {"class_type": "WanVideoTextEncode", "inputs": {"positive_prompt": shot_info["prompt"], "negative_prompt": shot_info["neg"], "t5": ["3", 0], "force_offload": True, "use_disk_cache": False, "device": "gpu"}},
        "6": {"class_type": "WanVideoModelLoader", "inputs": {"model": HIGH_MODEL, "base_precision": "bf16", "quantization": "disabled", "load_device": "offload_device", "attention_mode": "sdpa", "rms_norm_function": "default"}},
        "7": {"class_type": "WanVideoModelLoader", "inputs": {"model": LOW_MODEL, "base_precision": "bf16", "quantization": "disabled", "load_device": "offload_device", "attention_mode": "sdpa", "rms_norm_function": "default"}},
        "8": {"class_type": "WanVideoSampler", "inputs": {"model": ["6", 0], "image_embeds": ["4", 0], "steps": 4, "cfg": 1.0, "shift": 3.0, "seed": seed, "force_offload": True, "scheduler": "flowmatch_distill", "riflex_freq_index": 0, "text_embeds": ["5", 0], "start_step": 0, "end_step": 2, "denoise_strength": 1.0, "add_noise_to_samples": False, "batched_cfg": False, "rope_function": "comfy"}},
        "9": {"class_type": "WanVideoSampler", "inputs": {"model": ["7", 0], "image_embeds": ["4", 0], "steps": 4, "cfg": 1.0, "shift": 3.0, "seed": seed, "force_offload": True, "scheduler": "flowmatch_distill", "riflex_freq_index": 0, "text_embeds": ["5", 0], "samples": ["8", 0], "start_step": 2, "end_step": -1, "denoise_strength": 1.0, "add_noise_to_samples": False, "batched_cfg": False, "rope_function": "comfy"}},
        "10": {"class_type": "WanVideoDecode", "inputs": {"vae": ["2", 0], "samples": ["9", 0], "enable_vae_tiling": True, "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128, "normalization": "default"}},
        "11": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["10", 0], "frame_rate": shot_info["fps"], "loop_count": 0, "filename_prefix": shot_info["prefix"], "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": True, "pingpong": False, "save_output": True}},
    }

    result = post_json("/prompt", {"prompt": wf})
    prompt_id = result["prompt_id"]
    print(f"  queued {prompt_id}")
    history = wait_for_prompt(prompt_id)

    outputs = []
    for output in history.get("outputs", {}).values():
        for key in ("videos", "images"):
            for item in output.get(key, []):
                f = item.get("filename")
                if f:
                    outputs.append(OUTPUT_DIR / item.get("subfolder", "") / f)
    return outputs


def main():
    out_files = []
    for i, shot in enumerate(SHOTS):
        print(f"\n=== Shot {i+1}: {shot['prefix']} ===")
        seed = 222555 + i * 111
        files = generate_shot(shot, seed)
        out_files.extend(files)
        for f in files:
            print(f"  → {f}")

    print(f"\n=== Generated {len(out_files)} files ===")
    for f in out_files:
        print(f)


if __name__ == "__main__":
    main()
