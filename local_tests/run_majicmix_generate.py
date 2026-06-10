#!/usr/bin/env python3
"""ComfyUI image generation with Hires Fix (two-pass KSampler)."""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("/home/kang/Documents/ComfyUI/output")
CHECKPOINT = "majicmixRealistic_v7.safetensors"

DEFAULT_PROMPT = (
    "masterpiece, best quality, ultra detailed, photorealistic 3D Chinese manhua donghua style, "
    "breathtaking beautiful young woman, elegant oval face, delicate features, bright eyes, "
    "high nose bridge, soft lips, long glossy black hair, slim graceful posture, "
    "fitted white silk blouse, black high waist pencil skirt, diamond earrings, "
    "full body standing pose, modern city night window background, "
    "cinematic rim light, soft key light, shallow depth of field, 85mm lens, "
    "film still, clean skin, high fashion magazine quality"
)

DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, bad anatomy, bad face, deformed face, "
    "ugly, old, child, anime flat 2d, cartoon, doll, plastic skin, "
    "overexposed, underexposed, watermark, text, logo, extra fingers, bad hands, "
    "nsfw, nude"
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


def generate(
    prompt_text=DEFAULT_PROMPT,
    negative_text=DEFAULT_NEGATIVE,
    base_w=512,
    base_h=768,
    hires_w=960,
    hires_h=1280,
    steps=28,
    hires_steps=18,
    cfg=7.0,
    seed=42,
    filename_prefix="local_manhua/majicmix-comfy",
):
    """Two-pass generation: base 512x768 + latent upscale + KSampler refine."""

    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_text, "clip": ["1", 1]}},
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": base_w, "height": base_h, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "LatentUpscale",
            "inputs": {
                "samples": ["5", 0],
                "upscale_method": "bicubic",
                "width": hires_w,
                "height": hires_h,
                "crop": "disabled",
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["6", 0],
                "seed": seed + 1000,
                "steps": hires_steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 0.45,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": filename_prefix},
        },
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
    return outputs


if __name__ == "__main__":
    args = sys.argv[1:]
    prompt = args[0] if args else DEFAULT_PROMPT
    neg = args[1] if len(args) > 1 else DEFAULT_NEGATIVE
    bw = int(args[2]) if len(args) > 2 else 512
    bh = int(args[3]) if len(args) > 3 else 768
    hw = int(args[4]) if len(args) > 4 else 960
    hh = int(args[5]) if len(args) > 5 else 1280
    s1 = int(args[6]) if len(args) > 6 else 28
    s2 = int(args[7]) if len(args) > 7 else 18
    c = float(args[8]) if len(args) > 8 else 7.0
    sd = int(args[9]) if len(args) > 9 else 42
    generate(
        prompt_text=prompt, negative_text=neg,
        base_w=bw, base_h=bh, hires_w=hw, hires_h=hh,
        steps=s1, hires_steps=s2, cfg=c, seed=sd,
    )
