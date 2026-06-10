#!/usr/bin/env python3
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


SERVER = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("/home/kang/Documents/ComfyUI/output")
SOURCE = "/home/kang/Documents/ComfyUI/output/local_manhua/heroine-hq_00002_.png"


def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"ComfyUI rejected prompt: HTTP {exc.code}\n{body}") from exc


def get_json(path):
    with urllib.request.urlopen(SERVER + path, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_prompt(prompt_id, timeout_seconds=7200):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        history = get_json(f"/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(3)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def main():
    positive = (
        "A beautiful young woman in cream blouse and black pencil skirt, "
        "continuing to walk forward with natural confident strides on a city rooftop at golden hour, "
        "legs stepping forward one after another, arms swinging rhythmically, hair flowing in breeze, "
        "smooth continuous walking motion, cinematic warm sunset lighting, sharp clean image"
    )
    negative = (
        "static image, still frame, no movement, frozen pose, low quality, blurry, distorted face, "
        "deformed body, extra limbs, missing limbs, broken hands, ugly, bad anatomy, text, watermark, "
        "ghost, horror, flickering, morphing, morph, melting, grain, noise"
    )

    prompt = {
        "1": {"class_type": "VHS_LoadImagePath", "inputs": {"image": SOURCE, "custom_width": 480, "custom_height": 832}},
        "2": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": "split_files/vae/wan_2.1_vae.safetensors", "precision": "bf16", "use_cpu_cache": False, "verbose": False}},
        "3": {"class_type": "LoadWanVideoT5TextEncoder", "inputs": {"model_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "precision": "bf16", "load_device": "offload_device", "quantization": "disabled"}},
        "4": {"class_type": "WanVideoImageToVideoEncode", "inputs": {"width": 480, "height": 832, "num_frames": 65, "noise_aug_strength": 0.0, "start_latent_strength": 0.9, "end_latent_strength": 1.15, "force_offload": True, "vae": ["2", 0], "start_image": ["1", 0], "fun_or_fl2v_model": False, "tiled_vae": True, "augment_empty_frames": 0.0}},
        "5": {"class_type": "WanVideoTextEncode", "inputs": {"positive_prompt": positive, "negative_prompt": negative, "t5": ["3", 0], "force_offload": True, "use_disk_cache": False, "device": "gpu"}},
        "6": {"class_type": "WanVideoModelLoader", "inputs": {"model": "wan22-i2v-gguf/high_noise_260412/high_noise_260412/wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412-Q5_K_M.gguf", "base_precision": "bf16", "quantization": "disabled", "load_device": "offload_device", "attention_mode": "sdpa", "rms_norm_function": "default"}},
        "7": {"class_type": "WanVideoModelLoader", "inputs": {"model": "wan22-i2v-gguf/low_noise_260412/low_noise_260412/wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412-Q5_K_M.gguf", "base_precision": "bf16", "quantization": "disabled", "load_device": "offload_device", "attention_mode": "sdpa", "rms_norm_function": "default"}},
        "8": {"class_type": "WanVideoSampler", "inputs": {"model": ["6", 0], "image_embeds": ["4", 0], "steps": 4, "cfg": 1.0, "shift": 3.0, "seed": 222555, "force_offload": True, "scheduler": "flowmatch_distill", "riflex_freq_index": 6, "text_embeds": ["5", 0], "start_step": 0, "end_step": 2, "denoise_strength": 1.0, "add_noise_to_samples": False, "batched_cfg": False, "rope_function": "comfy"}},
        "9": {"class_type": "WanVideoSampler", "inputs": {"model": ["7", 0], "image_embeds": ["4", 0], "steps": 4, "cfg": 1.0, "shift": 3.0, "seed": 222555, "force_offload": True, "scheduler": "flowmatch_distill", "riflex_freq_index": 6, "text_embeds": ["5", 0], "samples": ["8", 0], "start_step": 2, "end_step": -1, "denoise_strength": 1.0, "add_noise_to_samples": False, "batched_cfg": False, "rope_function": "comfy"}},
        "10": {"class_type": "WanVideoDecode", "inputs": {"vae": ["2", 0], "samples": ["9", 0], "enable_vae_tiling": True, "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128, "normalization": "default"}},
        "11": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["10", 0], "frame_rate": 16, "loop_count": 0, "filename_prefix": "local_manhua/wan22-65f-16fps", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": True, "trim_to_audio": False, "pingpong": False, "save_output": True}},
    }

    result = post_json("/prompt", {"prompt": prompt})
    prompt_id = result["prompt_id"]
    print(f"queued prompt_id={prompt_id}")
    history = wait_for_prompt(prompt_id)

    outputs = []
    for output in history.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            for item in output.get(key, []):
                filename = item.get("filename")
                if filename:
                    outputs.append(OUTPUT_DIR / item.get("subfolder", "") / filename)
    if not outputs:
        print(json.dumps(history.get("outputs", {}), ensure_ascii=False, indent=2))
        raise SystemExit("No output files found")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
