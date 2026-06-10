#!/usr/bin/env python3
"""SDXL XXMix_9realisticSDXL generation - Chinese beauty specialist."""
import json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

SERVER = os.environ.get("COMFY_SERVER", "http://127.0.0.1:8188")
OUTPUT_DIR = Path(os.environ.get("COMFY_OUTPUT", os.path.expanduser("~/Documents/ComfyUI/output")))
CHECKPOINT = "XXMix_9realisticSDXL.safetensors"

PROMPT = (
    "masterpiece, best quality, 8k, 3D donghua CG render, genshin impact style, "
    "character design sheet, turnaround, three views, full body, clean crisp 3D rendering, "
    "front view: a stunningly gorgeous Chinese 3D anime heroine, urban drama style, "
    "elegant oval face, large expressive crystal clear eyes, delicate high nose, soft glossy lips, "
    "flawless smooth skin, light glamorous makeup, "
    "long silky black hair with soft waves and subtle highlights, "
    "wearing a chic tailored cream blazer, fitted black pencil skirt, nude heels, "
    "delicate gold necklace, luxury watch, "
    "side view: same character from right profile, elegant silhouette, "
    "back view: same character from behind, beautiful hair, blazer back detail, "
    "clean light gray studio background, professional character reference sheet, "
    "Chinese short drama 短剧 concept art, 崩坏星穹铁道 quality, cel-shaded 3D"
)
NEG = (
    "low quality, worst quality, blurry, ugly, deformed, bad anatomy, bad hands, "
    "old, child, fat, flat 2d anime, cartoon, plastic doll, overexposed, "
    "text, watermark, signature, logo, nsfw, nude, western face, European, Caucasian, "
    "different clothes each view, inconsistent character, hanfu, ancient, fantasy, sword, "
    "realistic photo, photography, raw photo, photorealistic, uncanny valley, dead eyes, "
    "cross eye, lazy eye, asymmetrical eyes, weird eyes, deformed eyes"
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


def wait(pid, t=3600):
    dl = time.time() + t
    while time.time() < dl:
        h = get_json(f"/history/{pid}")
        if pid in h: return h[pid]
        time.sleep(2)
    raise TimeoutError(pid)


def gen(seed=42, prefix="local_manhua/xxmix-cn"):
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {
            "class_type": "CLIPTextEncodeSDXL",
            "inputs": {"clip": ["1", 1], "text_g": PROMPT, "text_l": PROMPT,
                       "width": 1344, "height": 896, "crop_w": 0, "crop_h": 0,
                       "target_width": 1344, "target_height": 896},
        },
        "3": {
            "class_type": "CLIPTextEncodeSDXL",
            "inputs": {"clip": ["1", 1], "text_g": NEG, "text_l": NEG,
                       "width": 1344, "height": 896, "crop_w": 0, "crop_h": 0,
                       "target_width": 1344, "target_height": 896},
        },
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1344, "height": 896, "batch_size": 1}},
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
    h = wait(pid)
    out = []
    for o in h.get("outputs", {}).values():
        for img in o.get("images", []):
            f = img.get("filename")
            if f: out.append(OUTPUT_DIR / img.get("subfolder", "") / f)
    if not out: raise SystemExit("No output")
    for o in out: print(o)


if __name__ == "__main__":
    gen(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 42)
