#!/usr/bin/env python3
"""ComfyUI high-quality image generation - SD1.5 Hires Fix pushed to max."""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("/home/kang/Documents/ComfyUI/output")
CHECKPOINT = "majicmixRealistic_v7.safetensors"

PROMPT = (
    "masterpiece, best quality, photorealistic, "
    "beautiful young Chinese woman, cream silk blouse, black pencil skirt, "
    "full body dynamic walking pose mid-stride on modern city rooftop at golden hour, "
    "one leg stepping forward knee bent, opposite arm naturally swinging forward, "
    "long black hair blowing slightly in wind, confident graceful walking motion, "
    "elegant oval face, bright expressive eyes, flawless skin, "
    "warm cinematic rim lighting, soft natural key light, shallow depth of field, "
    "clean sharp focus, soft bokeh background, magazine quality"
)

NEGATIVE = (
    "low quality, worst quality, blurry, ugly, deformed face, bad anatomy, "
    "bad hands, extra fingers, missing fingers, child, old, fat, "
    "anime, cartoon, 2d, flat, plastic skin, doll, "
    "text, watermark, signature, logo, nsfw, nude, bad lighting"
)


def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"ComfyUI rejected: HTTP {exc.code}\n{body}") from exc


def get_json(path):
    with urllib.request.urlopen(SERVER + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_for_prompt(prompt_id, timeout_seconds=3600):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        history = get_json(f"/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 888999
    prefix = "local_manhua/heroine-hq"

    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 768, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["4", 0], "seed": seed, "steps": 40, "cfg": 6.5,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "LatentUpscale",
            "inputs": {"samples": ["5", 0], "upscale_method": "bicubic", "width": 960, "height": 1280, "crop": "disabled"},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["6", 0], "seed": seed + 1000, "steps": 25, "cfg": 6.0,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.38,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }

    result = post_json("/prompt", {"prompt": workflow})
    prompt_id = result["prompt_id"]
    print(f"queued prompt_id={prompt_id}")
    history = wait_for_prompt(prompt_id)

    outputs = []
    for output in history.get("outputs", {}).values():
        for item in output.get("images", []):
            f = item.get("filename")
            if f:
                outputs.append(OUTPUT_DIR / item.get("subfolder", "") / f)
    if not outputs:
        print(json.dumps(history.get("outputs", {}), ensure_ascii=False, indent=2))
        raise SystemExit("No output files found")
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
