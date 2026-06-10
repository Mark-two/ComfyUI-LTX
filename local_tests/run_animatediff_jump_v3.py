#!/usr/bin/env python3
"""
AnimateDiff+ControlNet: final attempt with aggressive settings.
Switches context window to 16 frames, denoise=1.0, ControlNet=1.3
"""
import json, sys, time, urllib.error, urllib.request
from pathlib import Path

SERVER = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("/home/kang/Documents/ComfyUI/output")
SOURCE = "/home/kang/Documents/novel-to-video/outputs/fullbody-heroine/20260607-203308-01.png"
SKELETON_DIR = "/home/kang/Documents/ComfyUI/input/jump_skeleton/20260607-204657"
CHECKPOINT = "majicmixRealistic_v7.safetensors"
MOTION_MODULE = "mm_sd_v15_v2.ckpt"
CONTROL_NET = "control_v11p_sd15_openpose_v2.pth" if Path("/home/kang/Documents/ComfyUI/models/controlnet/control_v11p_sd15_openpose_v2.pth").exists() else "control_v11p_sd15_openpose.pth"

# Try with 16 frames, context window = 16 for better temporal coherence
FRAMES = 16
WIDTH, HEIGHT = 512, 768
FPS = 8


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


def wait_for_prompt(prompt_id, timeout_seconds=3600):
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
        "cold CEO heroine, black hair, white blouse black skirt, jumping high in the air, "
        "dynamic midair pose, arms raised, solid light gray background, cinematic donghua render"
    )
    negative_text = (
        "low quality, blurry, bad anatomy, bad hands, extra limbs, missing limbs, "
        "deformed, ugly, child, flat anime, plastic doll, watermark, text, nsfw, nude, "
        "bad face, disfigured, mutation, different clothes, changing outfit, static pose, standing"
    )

    prompt = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": SOURCE},
        },
        "2": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT},
        },
        "3": {
            "class_type": "ADE_AnimateDiffLoaderGen1",
            "inputs": {
                "model": ["2", 0],
                "model_name": MOTION_MODULE,
                "beta_schedule": "sqrt_linear (AnimateDiff)",
                "context_options": ["20", 0],
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt_text, "clip": ["2", 1]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_text, "clip": ["2", 1]},
        },
        "6": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": CONTROL_NET},
        },
        "7": {
            "class_type": "VHS_LoadImagesPath",
            "inputs": {
                "directory": SKELETON_DIR,
                "image_load_cap": FRAMES,
                "skip_first_images": 0,
                "select_every_nth": 1,
            },
        },
        "8": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["5", 0],
                "control_net": ["6", 0],
                "image": ["7", 0],
                "strength": 1.3,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "9": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["1", 0], "vae": ["2", 2]},
        },
        "10": {
            "class_type": "RepeatLatentBatch",
            "inputs": {"samples": ["9", 0], "amount": FRAMES},
        },
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "positive": ["8", 0],
                "negative": ["8", 1],
                "latent_image": ["10", 0],
                "seed": 2606072901,
                "steps": 25,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 0.92,
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["2", 2]},
        },
        "13": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["12", 0],
                "frame_rate": FPS,
                "loop_count": 0,
                "filename_prefix": "local_manhua/jump-v3",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
        # Also save skeleton + keypoint comparison
        "14": {
            "class_type": "OpenposePreprocessor",
            "inputs": {"image": ["1", 0], "detect_hand": "disable", "detect_body": "enable", "detect_face": "disable"},
        },
        "15": {
            "class_type": "VHS_LoadImagesPath",
            "inputs": {
                "directory": SKELETON_DIR,
                "image_load_cap": 1,
                "skip_first_images": 8,
                "select_every_nth": 1,
            },
        },
        # Context options
        "20": {
            "class_type": "ADE_AnimateDiffUniformContextOptions",
            "inputs": {
                "context_length": 16,
                "context_stride": 1,
                "context_overlap": 4,
                "fuse_method": "pyramid",
                "context_schedule": "uniform",
                "closed_loop": False,
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

    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
