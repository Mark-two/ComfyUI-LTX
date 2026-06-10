#!/usr/bin/env python3
"""ComfyUI workflow: AnimateDiff + ControlNet OpenPose for jumping heroine."""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("/home/kang/Documents/ComfyUI/output")
SKELETON_DIR = "/home/kang/Documents/ComfyUI/input/jump_skeleton/20260607-204657"
CHECKPOINT = "majicmixRealistic_v7.safetensors"
MOTION_MODULE = "mm_sd_v15_v2.ckpt"
CONTROL_NET = "control_v11p_sd15_openpose.pth"

FRAMES = 32
WIDTH, HEIGHT = 512, 768
FPS = 12


def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER + path, data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"ComfyUI rejected: HTTP {exc.code}\n{body}") from exc


def get_json(path):
    with urllib.request.urlopen(SERVER + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_for_prompt(prompt_id, timeout_seconds=1800):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        history = get_json(f"/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def main():
    prompt_text = (
        "masterpiece, best quality, 3D Chinese manhua, full body elegant young woman, "
        "cold CEO heroine, black hair, white blouse black skirt, jumping dynamic action pose, "
        "solid light gray background, cinematic donghua render, sharp lighting, smooth motion"
    )
    negative_text = (
        "low quality, blurry, bad anatomy, bad hands, extra limbs, missing limbs, "
        "deformed, ugly, child, flat anime, plastic doll, watermark, text, nsfw, nude, "
        "bad face, disfigured, mutation"
    )

    prompt = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT},
        },
        "2": {
            "class_type": "ADE_AnimateDiffLoaderGen1",
            "inputs": {
                "model": ["1", 0],
                "model_name": MOTION_MODULE,
                "beta_schedule": "sqrt_linear (AnimateDiff)",
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt_text, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_text, "clip": ["1", 1]},
        },
        "5": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": CONTROL_NET},
        },
        "6": {
            "class_type": "VHS_LoadImagesPath",
            "inputs": {
                "directory": SKELETON_DIR,
                "image_load_cap": FRAMES,
                "skip_first_images": 0,
                "select_every_nth": 1,
            },
        },
        "7": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["3", 0],
                "negative": ["4", 0],
                "control_net": ["5", 0],
                "image": ["6", 0],
                "strength": 0.9,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "8": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": FRAMES},
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["7", 0],
                "negative": ["7", 1],
                "latent_image": ["8", 0],
                "seed": 2606072801,
                "steps": 25,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["9", 0], "vae": ["1", 2]},
        },
        "11": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["10", 0],
                "frame_rate": FPS,
                "loop_count": 0,
                "filename_prefix": "local_manhua/jump-animatediff",
                "format": "video/h264-mp4",
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
        for key in ("videos", "images"):
            for item in output.get(key, []):
                filename = item.get("filename")
                if filename:
                    outputs.append(OUTPUT_DIR / item.get("subfolder", "") / filename)

    if not outputs:
        print(json.dumps(history.get("outputs", {}), ensure_ascii=False, indent=2))
        raise SystemExit("No output files found")

    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
