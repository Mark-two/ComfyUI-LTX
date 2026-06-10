#!/usr/bin/env python3
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


SERVER = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("/home/kang/Documents/ComfyUI/output")


def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(path):
    with urllib.request.urlopen(SERVER + path, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_prompt(prompt_id, timeout_seconds=600):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        history = get_json(f"/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def main():
    positive = (
        "best quality, ultra detailed, cinematic realistic Chinese short drama heroine, "
        "modern elegant office lady, beautiful face, clear eyes, refined makeup, "
        "long black hair, fitted white blouse, black pencil skirt, subtle luxury, "
        "upper body portrait, confident expression, city night window background, "
        "dramatic rim light, film still, shallow depth of field"
    )
    negative = (
        "low quality, blurry, bad anatomy, bad hands, extra fingers, deformed face, "
        "cross-eye, ugly, old, child, anime, cartoon, watermark, text, logo, nsfw"
    )

    prompt = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "majicmixRealistic_v7.safetensors"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": positive},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": negative},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 768, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": 2606071701,
                "steps": 28,
                "cfg": 7.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "local_manhua/majicmix_keyframe"},
        },
    }

    try:
        result = post_json("/prompt", {"prompt": prompt})
    except urllib.error.URLError as exc:
        raise SystemExit(f"ComfyUI is not reachable at {SERVER}: {exc}") from exc

    prompt_id = result["prompt_id"]
    print(f"queued prompt_id={prompt_id}")
    history = wait_for_prompt(prompt_id)

    images = []
    for output in history.get("outputs", {}).values():
        for image in output.get("images", []):
            path = OUTPUT_DIR / image.get("subfolder", "") / image["filename"]
            images.append(path)

    if not images:
        raise SystemExit("No image output found in ComfyUI history")

    for image in images:
        print(image)


if __name__ == "__main__":
    main()
