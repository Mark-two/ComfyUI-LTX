#!/usr/bin/env python3
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


SERVER = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("/home/kang/Documents/ComfyUI/output")
SOURCE = "/home/kang/Documents/novel-to-video/outputs/fullbody-heroine/20260607-203308-01.png"


def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
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
    prompt_text = (
        "A full body 3D Chinese manhua heroine, elegant young woman with black hair, "
        "white blouse and black skirt, she jumps upward from the ground, dynamic midair motion, "
        "arms lifted naturally, skirt and hair moving with momentum, clean light gray studio background, "
        "cinematic lighting, high quality, stable face, consistent outfit"
    )

    prompt = {
        "1": {
            "class_type": "VHS_LoadImagePath",
            "inputs": {"image": SOURCE, "custom_width": 512, "custom_height": 768},
        },
        "2": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": "llava_llama3_vision.safetensors"},
        },
        "3": {
            "class_type": "CLIPVisionEncode",
            "inputs": {"clip_vision": ["2", 0], "image": ["1", 0], "crop": "center"},
        },
        "4": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "llava_llama3_fp16.safetensors",
                "type": "hunyuan_video",
            },
        },
        "5": {
            "class_type": "TextEncodeHunyuanVideo_ImageToVideo",
            "inputs": {
                "clip": ["4", 0],
                "clip_vision_output": ["3", 0],
                "prompt": prompt_text,
                "image_interleave": 2,
            },
        },
        "6": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "hyvideo/hunyuan_video_I2V_fp8_e4m3fn.safetensors",
                "weight_dtype": "default",
            },
        },
        "7": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "hyvid/hunyuan_video_vae_bf16.safetensors"},
        },
        "8": {
            "class_type": "HunyuanImageToVideo",
            "inputs": {
                "positive": ["5", 0],
                "vae": ["7", 0],
                "width": 512,
                "height": 768,
                "length": 33,
                "batch_size": 1,
                "guidance_type": "v2 (replace)",
                "start_image": ["1", 0],
            },
        },
        "9": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["6", 0], "conditioning": ["8", 0]},
        },
        "10": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": 2606080918},
        },
        "11": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "12": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["6", 0], "scheduler": "simple", "steps": 16, "denoise": 1.0},
        },
        "13": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["10", 0],
                "guider": ["9", 0],
                "sampler": ["11", 0],
                "sigmas": ["12", 0],
                "latent_image": ["8", 1],
            },
        },
        "14": {
            "class_type": "VAEDecodeTiled",
            "inputs": {"samples": ["13", 0], "vae": ["7", 0], "tile_size": 256, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8},
        },
        "15": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["14", 0],
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": "local_manhua/hunyuan-i2v-jump",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
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
