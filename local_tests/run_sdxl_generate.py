#!/usr/bin/env python3
"""ComfyUI SDXL image generation with Hires Fix."""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("/home/kang/Documents/ComfyUI/output")
CHECKPOINT = "dreamshaper-xl-1-0/unet/diffusion_pytorch_model.safetensors"

# SDXL needs a proper diffusers-style loader. Let's use the directory format.
# Actually, dreamshaper is stored as diffusers dir. CheckpointLoaderSimple
# may not pick it up. Let me check what formats are available.
# The checkpoint list should show what ComfyUI can load.


def get_available_checkpoints():
    import json as j
    with urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=30) as r:
        data = j.loads(r.read().decode("utf-8"))
    # get checkpoints from CheckpointLoaderSimple
    return data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]


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
    # Find available checkpoints
    ckpts = get_available_checkpoints()
    sdxl_ckpts = [c for c in ckpts if any(s in c.lower() for s in ['dreamshaper', 'sd_xl', 'sdxl'])]
    sd15_ckpts = [c for c in ckpts if 'majicmix' in c.lower()]
    
    print("SDXL checkpoints:", sdxl_ckpts)
    print("SD1.5 checkpoints:", sd15_ckpts)
    
    if sdxl_ckpts:
        ckpt = sdxl_ckpts[0]
        base_w, base_h = 1024, 1024
        hires_w, hires_h = 1280, 1664
        is_sdxl = True
        front_hint = ""
    elif sd15_ckpts:
        # fallback to majicMIX with higher resolution
        ckpt = sd15_ckpts[0]
        base_w, base_h = 512, 768
        hires_w, hires_h = 960, 1280
        is_sdxl = False
        front_hint = ""
    else:
        ckpt = CHECKPOINT
        base_w, base_h = 1024, 1024
        hires_w, hires_h = 1280, 1664
        is_sdxl = False
        front_hint = ""

    print(f"Using: {ckpt}, SDXL={is_sdxl}, {base_w}x{base_h}->{hires_w}x{hires_h}")
    
    # Updated prompt - more Chinese donghua style
    prompt_text = front_hint + (
        "masterpiece, best quality, ultra detailed, photorealistic, "
        "a breathtakingly beautiful young Chinese woman, elegant oval face, delicate refined features, "
        "large bright expressive eyes, high nose bridge, soft pink lips, flawless porcelain skin, "
        "long silky black hair flowing elegantly, graceful slender figure, "
        "wearing a luxurious cream silk blouse and fitted black pencil skirt, delicate diamond stud earrings, "
        "full body standing pose on a modern city rooftop at golden hour, "
        "warm cinematic rim lighting, soft natural key light, shallow depth of field, 85mm f1.4, "
        "clean sharp focus on face and eyes, soft bokeh background, magazine cover quality"
    )
    
    negative = (
        "low quality, worst quality, blurry, ugly, deformed face, bad anatomy, "
        "bad hands, extra fingers, missing fingers, fused fingers, too many fingers, "
        "child, old, fat, overweight, anime, cartoon, 2d, flat, plastic skin, doll, "
        "text, watermark, signature, logo, nsfw, nude, cleavage, bad lighting, underexposed, overexposed"
    )
    
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 777888
    prefix = "local_manhua/sdxl-heroine"

    if is_sdxl:
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "2": {"class_type": "CLIPTextEncodeSDXL", "inputs": {"text": prompt_text, "clip": ["1", 1], "width": base_w, "height": base_h, "crop_w": 0, "crop_h": 0, "target_width": base_w, "target_height": base_h}},
            "3": {"class_type": "CLIPTextEncodeSDXL", "inputs": {"text": negative, "clip": ["1", 1], "width": base_w, "height": base_h, "crop_w": 0, "crop_h": 0, "target_width": base_w, "target_height": base_h}},
        }
    else:
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        }

    # Common nodes
    workflow.update({
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": base_w, "height": base_h, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["4", 0], "seed": seed, "steps": 35, "cfg": 6.5,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "LatentUpscale",
            "inputs": {
                "samples": ["5", 0], "upscale_method": "bicubic",
                "width": hires_w, "height": hires_h, "crop": "disabled",
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["6", 0], "seed": seed + 1000, "steps": 22, "cfg": 6.0,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.42,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    })

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
