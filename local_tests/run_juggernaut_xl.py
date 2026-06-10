#!/usr/bin/env python3
"""SDXL Juggernaut XL image generation (clean, no FaceDetailer)."""
import json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

SERVER = os.environ.get("COMFY_SERVER", "http://127.0.0.1:8188")
OUTPUT_DIR = Path(os.environ.get("COMFY_OUTPUT", os.path.expanduser("~/Documents/ComfyUI/output")))
CHECKPOINT = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

SDXL_PROMPT = (
    "masterpiece, best quality, photorealistic, 8k, raw photo, "
    "a stunningly gorgeous glamorous woman, hourglass figure, full chest, slim waist, "
    "flawless glowing skin, captivating eyes, full lips, perfect makeup, "
    "long voluminous wavy blonde hair, confident seductive expression, "
    "wearing a luxurious tight burgundy evening gown, diamond necklace, "
    "full body standing in a luxury penthouse at night, city lights through window, "
    "warm golden rim lighting, soft key light, shallow depth of field, bokeh, "
    "sharp focus on face and body, professional fashion photography, "
    "Vogue editorial quality, high fashion, glamorous"
)
SDXL_NEG = (
    "low quality, worst quality, blurry, ugly, deformed face, bad anatomy, bad hands, "
    "extra fingers, missing fingers, child, old, fat, anime, cartoon, 2d, doll, plastic, "
    "text, watermark, signature, logo, nsfw, nude, bad lighting, "
    "flat chest, skinny, androgynous, masculine"
)

def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}\n{exc.read().decode('utf-8', errors='replace')}") from exc

def get_json(path):
    with urllib.request.urlopen(SERVER + path, timeout=30) as r: return json.loads(r.read().decode("utf-8"))

def wait_for_prompt(pid, timeout=3600):
    dl = time.time() + timeout
    while time.time() < dl:
        h = get_json(f"/history/{pid}")
        if pid in h: return h[pid]
        time.sleep(2)
    raise TimeoutError(pid)

def generate(seed=42, prefix="local_manhua/sdxl-heroine"):
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {
            "class_type": "CLIPTextEncodeSDXL",
            "inputs": {"clip": ["1", 1], "text_g": SDXL_PROMPT, "text_l": SDXL_PROMPT,
                       "width": 1024, "height": 1536, "crop_w": 0, "crop_h": 0,
                       "target_width": 1024, "target_height": 1536},
        },
        "3": {
            "class_type": "CLIPTextEncodeSDXL",
            "inputs": {"clip": ["1", 1], "text_g": SDXL_NEG, "text_l": SDXL_NEG,
                       "width": 1024, "height": 1536, "crop_w": 0, "crop_h": 0,
                       "target_width": 1024, "target_height": 1536},
        },
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 896, "height": 1344, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                       "latent_image": ["4", 0], "seed": seed, "steps": 35, "cfg": 6.5,
                       "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0},
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }
    r = post_json("/prompt", {"prompt": wf})
    pid = r["prompt_id"]
    print(f"pid={pid}")
    h = wait_for_prompt(pid)
    out = []
    for o in h.get("outputs", {}).values():
        for img in o.get("images", []):
            f = img.get("filename")
            if f: out.append(OUTPUT_DIR / img.get("subfolder", "") / f)
    if not out: raise SystemExit("No output")
    for o in out: print(o)

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    generate(seed=seed)
